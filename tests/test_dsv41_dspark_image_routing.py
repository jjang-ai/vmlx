"""DSV4.1 image requests take DSpark when the installed runner can prefill them; everything else keeps AR."""
import types

from vmlx_engine.utils.dsv41_dspark_generator import DSV41DSparkGenerator as G


class Runner:
    def prefill(self, prompt_ids, session=None, on_boundary=None, chunk_size=None, clear_cache=True, cuts=None,
                on_cut=None, images=None, token_types=None):
        raise NotImplementedError


class OldRunner:
    def prefill(self, prompt_ids, session=None, on_boundary=None, chunk_size=None, clear_cache=True, cuts=None,
                on_cut=None):
        raise NotImplementedError


def _gen(runner):
    g = G.__new__(G)
    g._runner = runner
    g._spec_sampler = lambda req: (True, None)
    return g


def _req(**kw):
    base = dict(pixel_values=None, pixel_values_videos=None, dsv41_vl_images=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_text_request_keeps_dspark():
    assert _gen(Runner())._use_dspark(_req()) is True


def test_image_request_takes_dspark_with_a_capable_runner(monkeypatch):
    monkeypatch.delenv("VMLX_DSV41_DSPARK_VL", raising=False)
    assert _gen(Runner())._use_dspark(_req(dsv41_vl_images=[object()])) is True


def test_image_request_keeps_ar_with_an_older_runner(monkeypatch):
    monkeypatch.delenv("VMLX_DSV41_DSPARK_VL", raising=False)
    assert _gen(OldRunner())._use_dspark(_req(dsv41_vl_images=[object()])) is False


def test_diagnostic_lever_restores_ar(monkeypatch):
    monkeypatch.setenv("VMLX_DSV41_DSPARK_VL", "0")
    assert _gen(Runner())._use_dspark(_req(dsv41_vl_images=[object()])) is False


def test_other_vision_inputs_keep_ar(monkeypatch):
    monkeypatch.delenv("VMLX_DSV41_DSPARK_VL", raising=False)
    assert _gen(Runner())._use_dspark(_req(pixel_values=object())) is False
    assert _gen(Runner())._use_dspark(_req(pixel_values_videos=object())) is False


def test_no_runner_keeps_ar():
    assert _gen(None)._use_dspark(_req(dsv41_vl_images=[object()])) is False
