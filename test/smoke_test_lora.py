import copy
import random

import torch

from f5_tts.model import CFM, DiT
from f5_tts.model.lora import (
    EMBED_KEY,
    LoRALinear,
    count_params,
    expand_embedding_rows,
    inject_lora,
    load_trainable_state_dict,
    merge_and_unload,
    param_groups,
    set_trainable,
    trainable_state_dict,
    val_loss_sum,
)

MEL = dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100, target_sample_rate=24000, mel_spec_type="vocos")
CHARS = " abcdefghijklmnop"
VOCAB = {c: i for i, c in enumerate(CHARS)}
DEPTH = 2


def tiny():
    torch.manual_seed(1)
    return CFM(
        transformer=DiT(dim=64, depth=DEPTH, heads=2, dim_head=32, ff_mult=2, text_dim=32, conv_layers=2,
                        text_num_embeds=len(VOCAB), mel_dim=100),
        mel_spec_kwargs=MEL, vocab_char_map=VOCAB,
    )


def loss_of(model, seed=0):
    """CFM.forward draws random time / noise / masks: seed everything so two models see the same draws."""
    torch.manual_seed(seed)
    random.seed(seed)
    mel = torch.randn(2, 40, 100)
    loss, _, _ = model(mel, text=["abc a", "bca"], lens=torch.tensor([40, 30]))
    return loss


def main():
    base = tiny().eval()
    keys_before = set(base.state_dict().keys())
    loss_base = loss_of(base)

    # 1) injection: right number of layers, and the model output is unchanged at init (B = 0)
    model = copy.deepcopy(base)
    names = inject_lora(model, "attn,ff", r=4, alpha=8, dropout=0.0)
    assert len(names) == DEPTH * 6, f"expected {DEPTH * 6} LoRA layers, got {len(names)}: {names}"
    assert all(isinstance(model.get_submodule(n), LoRALinear) for n in names)
    assert torch.allclose(loss_of(model.eval()), loss_base, atol=1e-6), "LoRA must not change the output at init"
    print("1) injection OK:", len(names), "layers, output identical at init")

    # 2) only LoRA + text params are trainable
    set_trainable(model, train_text=True)
    train_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert train_names and all(("lora_A" in n or "lora_B" in n or "text_embed" in n) for n in train_names), train_names
    assert not any("transformer_blocks" in n and ".base." in n for n in train_names)
    total, trainable = count_params(model)
    print(f"2) trainable selection OK: {trainable} of {total} params")

    # 3) one real step: grads reach LoRA + text embedding, never the frozen base weights
    model.train()
    loss_of(model).backward()
    pm = dict(model.named_parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in pm.items() if "lora_B" in n), "no grad on lora_B"
    assert pm[EMBED_KEY].grad is not None, "no grad on the text embedding"
    assert all(p.grad is None for n, p in pm.items() if ".base." in n), "frozen base weight received a gradient"
    groups = param_groups(model, 1e-4, 2e-4)
    assert len(groups) == 2 and groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 2e-4
    print("3) backward OK: grads only on LoRA + text embedding")

    # 4) merge == LoRA model, and the original key structure is restored
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn_like(p) * 0.05)
    model.eval()
    loss_lora = loss_of(model)
    sd_trainable = trainable_state_dict(model)
    merged = merge_and_unload(copy.deepcopy(model)).eval()
    loss_merged = loss_of(merged)
    assert torch.allclose(loss_lora, loss_merged, atol=1e-5), (loss_lora.item(), loss_merged.item())
    assert not torch.allclose(loss_lora, loss_base, atol=1e-6), "LoRA had no effect after perturbing B"
    assert set(merged.state_dict().keys()) == keys_before, "merged model must have the stock key names"
    print(f"4) merge OK: loss {loss_lora.item():.5f} == {loss_merged.item():.5f}; stock key names restored")

    # 5) save / load of the trainable parameters
    fresh = copy.deepcopy(base)
    inject_lora(fresh, "attn,ff", r=4, alpha=8, dropout=0.0)
    set_trainable(fresh, train_text=True)
    load_trainable_state_dict(fresh, sd_trainable)
    assert torch.allclose(loss_of(fresh.eval()), loss_lora, atol=1e-6)
    print("5) trainable state dict round-trip OK")

    # 6) vocabulary extension keeps the old rows
    w = base.state_dict()[EMBED_KEY]
    w2 = expand_embedding_rows(w, w.shape[0] + 5)
    assert w2.shape == (w.shape[0] + 5, w.shape[1]) and torch.equal(w2[: w.shape[0]], w)
    print("6) embedding expansion OK:", tuple(w.shape), "->", tuple(w2.shape))

    # 7) validation loss: deterministic, and it does not disturb the global RNG (training must be unaffected)
    batch = {"mel": torch.randn(2, 100, 40), "text": ["abc a", "bca"], "mel_lengths": torch.tensor([40, 30])}
    model.eval()
    torch.manual_seed(5)
    before = torch.rand(3)
    torch.manual_seed(5)
    s1, n1 = val_loss_sum(model, [batch], "cpu")
    after = torch.rand(3)
    s2, n2 = val_loss_sum(model, [batch], "cpu")
    assert n1 == n2 == 1 and abs(s1 - s2) < 1e-7, f"validation loss must be repeatable: {s1} vs {s2}"
    assert torch.equal(before, after), "val_loss_sum must restore the global RNG state"
    print(f"7) validation loss OK: repeatable ({s1:.5f}) and RNG state restored")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()