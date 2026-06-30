from types import SimpleNamespace


def test_smelt_load_applies_internal_jang_norm_shift(monkeypatch, tmp_path):
    import jang_tools.loader as jt_loader
    import vmlx_engine.utils.jang_loader as internal_jang_loader
    import vmlx_engine.utils.smelt_loader as smelt_loader

    calls = []

    class FakeModel:
        def __init__(self):
            self.model = SimpleNamespace(layers=[])

        def parameters(self):
            return {}

    fake_model = FakeModel()
    fake_tokenizer = object()

    monkeypatch.setattr(
        jt_loader,
        "load_jang_model",
        lambda path: (fake_model, fake_tokenizer),
    )
    monkeypatch.setattr(
        jt_loader,
        "_fix_quantized_bits",
        lambda model, overrides: calls.append(("fix_bits", model, overrides)),
    )
    monkeypatch.setattr(
        internal_jang_loader,
        "apply_jang_norm_shift",
        lambda model: calls.append(("norm_shift", model)),
    )
    monkeypatch.setattr(
        smelt_loader.ExpertIndex,
        "build",
        staticmethod(lambda path: SimpleNamespace(num_experts=8, layers={})),
    )
    monkeypatch.setattr(smelt_loader.mx, "eval", lambda *args, **kwargs: None)
    monkeypatch.setattr(smelt_loader.mx, "clear_cache", lambda: None)

    model, tokenizer = smelt_loader.smelt_load(str(tmp_path), expert_percent=50)

    assert model is fake_model
    assert tokenizer is fake_tokenizer
    assert ("fix_bits", fake_model, {}) in calls
    assert ("norm_shift", fake_model) in calls
