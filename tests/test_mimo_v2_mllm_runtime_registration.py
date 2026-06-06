import sys
import types


def test_registered_mimo_v2_mllm_model_forwards_inputs_embeds(monkeypatch):
    from vmlx_engine.models import mllm

    module_name = "mlx_vlm.models.mimo_v2"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    fake_runtime = types.SimpleNamespace()

    class FakeTextConfig:
        quantization = None

        @classmethod
        def from_dict(cls, params):
            return cls()

    class FakeTextModel:
        def __init__(self, config):
            self.config = config
            self.calls = []
            self.model = types.SimpleNamespace(embed_tokens=lambda input_ids: "embeds")
            self.layers = []

        def make_cache(self):
            return []

        def sanitize(self, weights):
            return weights

        def load_weights(self, weights, strict=True):
            return None

        def __call__(
            self,
            input_ids=None,
            *,
            inputs_embeds=None,
            cache=None,
            mask=None,
            **kwargs,
        ):
            self.calls.append(
                {
                    "input_ids": input_ids,
                    "inputs_embeds": inputs_embeds,
                    "cache": cache,
                    "mask": mask,
                    "kwargs": kwargs,
                }
            )
            return "logits"

    fake_runtime.ModelArgs = FakeTextConfig
    fake_runtime.Model = FakeTextModel

    real_import_module = mllm.importlib.import_module

    def fake_import_module(name):
        if name == "jang_tools.mimo_v2.mlx_model":
            return fake_runtime
        return real_import_module(name)

    monkeypatch.setattr(mllm.importlib, "import_module", fake_import_module)

    mllm._register_mimo_v2_mlx_vlm_runtime()
    registered = sys.modules[module_name]
    model = registered.Model(registered.ModelConfig.from_dict({"model_type": "mimo_v2"}))

    result = model(
        "token-ids",
        inputs_embeds="provided-embeds",
        cache="cache-object",
        mask="mask-object",
        custom_kwarg="kept",
    )

    assert result.logits == "logits"
    assert model.language_model.inner.calls == [
        {
            "input_ids": "token-ids",
            "inputs_embeds": "provided-embeds",
            "cache": "cache-object",
            "mask": "mask-object",
            "kwargs": {"custom_kwarg": "kept"},
        }
    ]
