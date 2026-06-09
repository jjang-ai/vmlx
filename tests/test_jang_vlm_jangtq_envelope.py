def test_jangtq_mxtq_vlm_envelope_is_accepted(tmp_path, monkeypatch):
    from vmlx_engine.utils import jang_loader

    model_dir = tmp_path / "vl-jangtq"
    model_dir.mkdir()
    (model_dir / "jang_config.json").write_text(
        '{"version":2,"format":"jangtq","weight_format":"mxtq",'
        '"mxtq_bits":{"routed_expert":2}}'
    )
    (model_dir / "config.json").write_text('{"model_type":"qwen2_vl"}')

    sentinel = object()
    monkeypatch.setattr(jang_loader, "_is_v2_model", lambda path: True)
    monkeypatch.setattr(
        jang_loader, "_load_jang_v2_vlm", lambda *args, **kwargs: sentinel
    )
    monkeypatch.setattr(
        jang_loader, "_ensure_zaya_runtime_supported", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        jang_loader,
        "_ensure_jang_family_runtime_supported",
        lambda *args, **kwargs: None,
    )

    assert jang_loader.load_jang_vlm_model(model_dir) is sentinel


def test_jangtq_format_only_vlm_envelope_is_accepted(tmp_path, monkeypatch):
    from vmlx_engine.utils import jang_loader

    model_dir = tmp_path / "vl-jangtq-format-only"
    model_dir.mkdir()
    (model_dir / "jang_config.json").write_text(
        '{"version":2,"format":"jangtq","profile":"JANGTQ_2",'
        '"tq_layout":"prestacked_switch_mlp"}'
    )
    (model_dir / "config.json").write_text('{"model_type":"mimo_v2"}')

    sentinel = object()
    monkeypatch.setattr(jang_loader, "_is_v2_model", lambda path: True)
    monkeypatch.setattr(
        jang_loader, "_load_jang_v2_vlm", lambda *args, **kwargs: sentinel
    )
    monkeypatch.setattr(
        jang_loader, "_ensure_zaya_runtime_supported", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        jang_loader,
        "_ensure_jang_family_runtime_supported",
        lambda *args, **kwargs: None,
    )

    assert jang_loader.load_jang_vlm_model(model_dir) is sentinel
