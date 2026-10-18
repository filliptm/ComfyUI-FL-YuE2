import torch
import pytest

from fl_yue2.yue2.model import attention
from fl_yue2.yue2.downloads import contained, resolve
from fl_yue2.yue2.protocol import token_prefixes, negative_prefix, GenerationConfig, ABC_END, MUSIC_START
from fl_yue2.yue2 import runtime


@pytest.mark.parametrize("causal", [False, True])
# length=4 equals the query head count: guards gqa detection against confusing
# the kv head axis with the sequence axis when the two happen to match.
@pytest.mark.parametrize("length", [1, 4, 7, 270])
def test_attention_preserves_gqa_and_causality(causal, length):
    generator = torch.Generator().manual_seed(10)
    q = torch.randn(1, length, 4, 8, generator=generator)
    k = torch.randn(1, length, 2, 8, generator=generator)
    v = torch.randn(1, length, 2, 8, generator=generator)
    expected = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                                enable_gqa=True, is_causal=causal).transpose(1, 2)
    torch.testing.assert_close(attention(q, k, v, causal), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("length", [1, 7, 270])
def test_attention_never_uses_device_selected_backend(monkeypatch, causal, length):
    # xformers on RTX 50x (sm_120) has no operator for a dense tensor attn_bias, so
    # attention() must bypass ComfyUI's device-selected backend entirely; every call
    # goes through comfy.ops.scaled_dot_product_attention, which dispatches to the
    # flash/cudnn/efficient backends that support sm_120.
    import fl_yue2.yue2.model as model_module

    # The fix is the absence of the device-selected backend: model.py must not
    # even import it (xformers dispatch on sm_120 has no operator for a dense
    # tensor attn_bias), so assert that first.
    assert not hasattr(model_module, "optimized_attention_for_device")

    def broken_for_device(device, mask=False, small_input=False):
        raise NotImplementedError("No operator found for `memory_efficient_attention_forward` "
                                  "(simulated sm_120 xformers dispatch)")

    # Defense in depth: if any path reaches the backend through the comfy
    # module instead of a local import, fail loudly here.
    import comfy.ldm.modules.attention as comfy_attention
    monkeypatch.setattr(comfy_attention, "optimized_attention_for_device", broken_for_device)
    generator = torch.Generator().manual_seed(10)
    q = torch.randn(1, length, 4, 8, generator=generator)
    k = torch.randn(1, length, 2, 8, generator=generator)
    v = torch.randn(1, length, 2, 8, generator=generator)

    calls = []
    real_sdpa = model_module.comfy.ops.scaled_dot_product_attention

    def spy(q_, k_, v_, *args, **kwargs):
        calls.append(kwargs.get("attn_mask") is not None or kwargs.get("is_causal", False))
        return real_sdpa(q_, k_, v_, *args, **kwargs)

    monkeypatch.setattr(model_module.comfy.ops, "scaled_dot_product_attention", spy)
    expected = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                                enable_gqa=True, is_causal=causal).transpose(1, 2)
    try:
        got = model_module.attention(q, k, v, causal)
    finally:
        monkeypatch.undo()
    torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-5)
    assert calls, "attention() must route through comfy.ops.scaled_dot_product_attention"


def test_download_boundary(tmp_path):
    with pytest.raises(ValueError):
        contained(tmp_path, "../outside")
    with pytest.raises(ValueError):
        resolve("../../outside", False)


def test_offline_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.resolve.__globals__["folder_paths"], "get_folder_paths", lambda _: [str(tmp_path)])
    with pytest.raises(FileNotFoundError):
        resolve("YuE2-3B", False)
    assert not (tmp_path / "YuE2-3B").exists()


def test_external_score_and_direct_do_not_load_model():
    class Tokenizer:
        def encode(self, text):
            return list(text.encode())
    model = runtime.MusicModel(None, Tokenizer(), None)
    supplied = runtime.make_plan(model, "pop", "hello", 42, "melody", "X:1\nK:C\nCDEF|", 128)
    assert supplied.abc_ids == Tokenizer().encode(supplied.abc)
    assert supplied.prefix == token_prefixes(supplied.request, model.tokenizer, supplied.abc_ids)
    direct = runtime.make_plan(model, "pop", "hello", 42, "off", "", 128)
    assert direct.abc is None and direct.abc_ids == []
    with pytest.raises(ValueError):
        runtime.make_plan(model, "pop", "hello", 42, "off", "X:1", 128)


@pytest.mark.parametrize("mode", ["full", "melody", "off"])
@pytest.mark.parametrize("lyrics", ["", " \n\t"])
def test_blank_lyrics_respect_planning(mode, lyrics, monkeypatch):
    class Tokenizer:
        def encode(self, text):
            return list(text.encode())
        def decode(self, ids):
            return bytes(ids).decode()
    calls = []
    score = "X:1\nK:C\nCDEF|"
    music = runtime.MusicModel(None, Tokenizer(), GenerationConfig())
    monkeypatch.setattr(music, "prepare", lambda tokens: calls.append(tokens))
    def generate(model, prefix, sampling, seed, phase, **kwargs):
        assert phase == "abc" and sampling.max_tokens == 4096
        return Tokenizer().encode(score), {}, False
    monkeypatch.setattr(runtime, "generate_tokens", generate)
    plan = runtime.make_plan(music, "instrumental trip hop", lyrics, 831001, mode, "", 4096)
    assert plan.request.cot == mode
    assert plan.request.lyrics == lyrics
    assert plan.abc == (None if mode == "off" else score)
    assert len(calls) == (0 if mode == "off" else 1)
    assert plan.prefix == token_prefixes(plan.request, music.tokenizer, plan.abc_ids)
    if mode != "off":
        supplied = runtime.make_plan(music, "instrumental", lyrics, 831001, mode, "X:1\nK:C\nCDEF|", 4096)
        assert supplied.request.cot == mode
        assert supplied.abc_ids
        assert len(calls) == 1


@pytest.mark.parametrize("mode", ["full", "melody"])
@pytest.mark.parametrize("lyrics", ["", "[Verse]\nSing these words"])
def test_score_limit_preserves_partial_plan(monkeypatch, mode, lyrics):
    class Tokenizer:
        def encode(self, text):
            return list(text.encode())
        def decode(self, ids):
            return bytes(ids).decode()
    music = runtime.MusicModel(None, Tokenizer(), GenerationConfig())
    partial = "X:1\nK:C\nCDEF|G"
    ids = music.tokenizer.encode(partial)
    monkeypatch.setattr(music, "prepare", lambda tokens: None)
    monkeypatch.setattr(runtime, "generate_tokens", lambda *args, **kwargs: (ids, {}, True))
    plan = runtime.make_plan(music, "pop", lyrics, 42, mode, "", 128)
    assert plan.truncated
    assert plan.request.cot == mode
    assert plan.request.style == "pop" and plan.request.lyrics == lyrics and plan.request.seed == 42
    assert plan.abc == partial and plan.abc_ids == ids
    assert plan.prefix == token_prefixes(plan.request, music.tokenizer, ids)
    assert plan.prefix[-len(ids)-2:] == ids + [ABC_END, MUSIC_START]
    negative = negative_prefix(plan.request, music.tokenizer, plan.abc_ids)
    assert negative[-len(ids)-2:] == ids + [ABC_END, MUSIC_START]


def test_interruption_propagates():
    runtime.mm.interrupt_current_processing(True)
    try:
        with pytest.raises(runtime.mm.InterruptProcessingException):
            runtime.cancelled()
    finally:
        runtime.mm.interrupt_current_processing(False)
