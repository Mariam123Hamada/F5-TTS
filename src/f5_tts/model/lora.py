import math 
import torch
import torch.nn as nn 
import re 

"""
This is sample from the Arc :
time_embed.time_mlp.0 : Linear(in_features=256, out_features=1024, bias=True)
time_embed.time_mlp.2 : Linear(in_features=1024, out_features=1024, bias=True)
input_embed.proj : Linear(in_features=240, out_features=1024, bias=True)
transformer_blocks.0.attn_norm.linear : Linear(in_features=1024, out_features=6144, bias=True)
transformer_blocks.0.attn.to_q : Linear(in_features=1024, out_features=1024, bias=True)
transformer_blocks.0.attn.to_k : Linear(in_features=1024, out_features=1024, bias=True)
transformer_blocks.0.attn.to_v : Linear(in_features=1024, out_features=1024, bias=True)
transformer_blocks.0.attn.to_out.0 : Linear(in_features=1024, out_features=1024, bias=True)
transformer_blocks.0.ff.ff.0.0 : Linear(in_features=1024, out_features=2048, bias=True)
transformer_blocks.0.ff.ff.2 : Linear(in_features=2048, out_features=1024, bias=True)
transformer_blocks.1.attn_norm.linear : Linear(in_features=1024, out_features=6144, bias=True)
"""

TARGET_PRESETS = {
    "attn": [r"transformer_blocks\.\d+\.attn\.to_q$", r"transformer_blocks\.\d+\.attn\.to_k$",
             r"transformer_blocks\.\d+\.attn\.to_v$", r"transformer_blocks\.\d+\.attn\.to_out\.0$"],
    "ff": [r"transformer_blocks\.\d+\.ff\.ff\.0\.0$", r"transformer_blocks\.\d+\.ff\.ff\.2$"],
    "adaln": [r"transformer_blocks\.\d+\.attn_norm\.linear$"],
}

TEXT_TRAIN_PATTERNS = [r"transformer\.text_embed\.text_embed\.", r"transformer\.text_embed\.text_blocks\."]
 

class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B(A(dropout(x)))   with W, b frozen."""
 
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))   # zeros -> starts exactly as the base model
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False
 
    def forward(self, x):
        out = self.base(x)
        delta = (self.dropout(x) @ self.lora_A.t().to(x.dtype)) @ self.lora_B.t().to(x.dtype)
        return out + delta * self.scaling

def _match(name, patterns):
    return any(re.search(p, name) for p in patterns)
 
 
def resolve_targets(targets):
    """'attn,ff' (or a list) -> list of regex patterns."""
    if isinstance(targets, str):
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    patterns = []
    for t in targets:
        if t not in TARGET_PRESETS:
            raise ValueError(f"unknown LoRA target '{t}', choose from {list(TARGET_PRESETS)}")
        patterns += TARGET_PRESETS[t]
    return patterns

def inject_lora(model: nn.Module, targets="attn,ff", r=16, alpha=32.0, dropout=0.0):
    """Replace matching nn.Linear modules by LoRALinear. Returns the list of replaced module names."""
    patterns = resolve_targets(targets)
    found = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear) and _match(name, patterns)]
    if not found:
        raise RuntimeError("no Linear layer matched the LoRA targets - wrong model?")
    for name, lin in found:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child] = LoRALinear(lin, r=r, alpha=alpha, dropout=dropout)
    return [n for n, _ in found]
 
def set_trainable(model: nn.Module, train_text: bool = True):
    """Freeze everything, then unfreeze LoRA matrices (+ text embedding / text blocks if train_text)."""
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
    if train_text:
        for name, p in model.named_parameters():
            if _match(name, TEXT_TRAIN_PATTERNS):
                p.requires_grad = True


def param_groups(model: nn.Module, lr_lora: float, lr_text: float):
    """Two groups so the (new) text embeddings can learn faster than the LoRA matrices."""
    lora, text = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (lora if ("lora_A" in name or "lora_B" in name) else text).append(p)
    groups = [{"params": lora, "lr": lr_lora}]
    if text:
        groups.append({"params": text, "lr": lr_text})
    return groups


def trainable_state_dict(model: nn.Module):
    """Only what training changed (small): LoRA matrices + fully-trained text parameters."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
 

def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train
 

def load_trainable_state_dict(model: nn.Module, sd: dict):
    params = dict(model.named_parameters())
    missing = [k for k in sd if k not in params]
    if missing:
        raise RuntimeError(f"{len(missing)} keys of the LoRA checkpoint do not exist in the model, e.g. {missing[:3]}")
    with torch.no_grad():
        for k, v in sd.items():
            params[k].copy_(v.to(params[k].dtype))
    expected = {n for n, p in params.items() if p.requires_grad}
    not_in_ckpt = expected - set(sd)
    if not_in_ckpt:
        raise RuntimeError(f"{len(not_in_ckpt)} trainable params are missing in the LoRA checkpoint, e.g. {sorted(not_in_ckpt)[:3]}")
 
 
@torch.no_grad()
def merge_and_unload(model: nn.Module):
    """W <- W + scaling * B @ A, then put the plain nn.Linear back. Result = stock F5-TTS structure."""
    todo = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name, m in todo:
        delta = (m.lora_B.float() @ m.lora_A.float()) * m.scaling
        m.base.weight.add_(delta.to(m.base.weight.dtype))
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child] = m.base
        for p in m.base.parameters():
            p.requires_grad = False
    return model

EMBED_KEY = "transformer.text_embed.text_embed.weight"
@torch.no_grad()
def expand_embedding_rows(weight: torch.Tensor, new_rows: int, seed: int = 0):
    """weight: (old_rows, dim) -> (new_rows, dim). Old rows are kept (same indices); new rows start near the
    mean of the old ones (+ small noise) instead of pure random, which makes early training more stable."""
    old_rows, dim = weight.shape
    if new_rows < old_rows:
        raise ValueError(f"new_rows ({new_rows}) < old rows ({old_rows})")
    if new_rows == old_rows:
        return weight
    g = torch.Generator().manual_seed(seed)
    w = weight.float()
    extra = w.mean(0, keepdim=True) + 0.5 * w.std() * torch.randn(new_rows - old_rows, dim, generator=g)
    return torch.cat([weight, extra.to(weight.dtype)], dim=0)  