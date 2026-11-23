import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import InputEmbeddingsFeatures
from .primitives import _split_gate_up, _to_mlx_conv, masked_scatter
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


_MLX_EXPORT_LAYER_KEYS = {
    "mlp.gate.weight": "mlp.gate_weight",
    "mlp.gate.e_score_correction_bias": "mlp.e_score_correction_bias",
    "post_attention_convolution.weight": "attn_sconv.conv.weight",
    "post_mlp_convolution.weight": "mlp_sconv.conv.weight",
    "self_attn.k_sconv.weight": "self_attn.k_sconv.conv.weight",
    "self_attn.v_sconv.weight": "self_attn.v_sconv.conv.weight",
    "self_attn.q_l2_clip.scale": "self_attn.q_scale",
    "self_attn.rel_l2_clip.scale": "self_attn.rel_scale",
    "self_attn.rel_logits_proj.proj": "self_attn.rel_proj",
}
for _source, _target in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
    for _projection, _weight in (
        ("down_proj", "down_weight"),
        ("forget_gate_proj", "forget_weight"),
        ("message_gate_proj", "gain_weight"),
    ):
        _MLX_EXPORT_LAYER_KEYS[f"{_source}_message_gate.{_projection}.weight"] = (
            f"controllers.{_target}.{_weight}"
        )

_MLX_EXPORT_VISION_KEYS = {
    "vision_tower.embeddings.position_embedding.weight": "vision_tower.embeddings.position_embedding",
    "vision_tower.merger.linear_1.weight": "vision_tower.proj_in.weight",
    "vision_tower.merger.linear_2.weight": "vision_tower.proj_out.weight",
}


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        config.vision_config.text_hidden_size = config.text_config.hidden_size
        config.vision_config.out_hidden_size = config.text_config.hidden_size
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionModel(config.vision_config)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        h = self.language_model.model.embed(input_ids)
        if pixel_values is not None:
            features = self.vision_tower(
                pixel_values, image_grid_thw=kwargs.get("image_grid_thw")
            ).astype(h.dtype)
            slots = input_ids == self.config.image_token_id
            if int(slots.sum()) != features.shape[0]:
                raise ValueError("Image tokens do not match image features")
            h = masked_scatter(h, mx.broadcast_to(slots[..., None], h.shape), features)
        return InputEmbeddingsFeatures(inputs_embeds=h)

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        h = self.get_input_embeddings(input_ids, pixel_values, **kwargs).inputs_embeds
        return self.language_model(inputs_embeds=h, cache=cache, **kwargs)

    def make_cache(self):
        return self.language_model.make_cache()

    def shard(self, group=None):
        self.language_model.shard(group)

    @property
    def layers(self):
        return self.language_model.layers

    def quantization_from_metadata(self, metadata):
        if metadata.get("quant_method") != "inkling2":
            return None
        result = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
        for group in metadata["model"]["parameter_groups"]:
            kind = group.get("weight_transform")
            if kind == "cast":
                # Safetensors already stores these groups in their declared
                # FP32/BF16 dtypes; loading must not apply a global BF16 cast.
                # The unembedding's FP32 forward output is handled by the LM.
                continue
            if kind not in ("nvfp4_nearest", "mxfp8_nearest"):
                raise ValueError(f"Unsupported Inkling 2 weight transform: {kind}")
            settings = (
                {"group_size": 16, "bits": 4, "mode": "nvfp4"}
                if kind == "nvfp4_nearest"
                else {"group_size": 32, "bits": 8, "mode": "mxfp8"}
            )
            for tensor in group["tensors"]:
                layer = tensor.split("layers.")[1].split(".")[0]
                projections = (
                    ("gate_proj", "up_proj")
                    if tensor.endswith("w13_weight")
                    else ("down_proj",)
                )
                for projection in projections:
                    result[
                        f"language_model.model.layers.{layer}.mlp.switch_mlp.{projection}"
                    ] = settings.copy()
        return result

    @staticmethod
    def cast_predicate(path):
        # Preserve the native FP32 queries, routing bias, and NVFP4 outer
        # scales during conversion. Router weights may also arrive in FP32.
        return not any(
            s in path
            for s in (
                ".query",
                "message_queries",
                "gate_weight",
                "e_score_correction_bias",
                "weight_scale_2",
            )
        )

    def sanitize(self, weights):
        out, experts = {}, {}
        attn_names = {
            "wq_du": "q_proj",
            "wk_dv": "k_proj",
            "wv_dv": "v_proj",
            "wr_du": "r_proj",
            "wo_ud": "o_proj",
            "gate_attn": "gate_proj",
            "attn_sink_proj": "sink_proj",
        }
        norms = {
            "attn_norm": "input_layernorm",
            "mlp_norm": "post_attention_layernorm",
            "q_input_norm": "q_input_layernorm",
            "k_input_norm": "k_input_layernorm",
            "v_input_norm": "v_input_layernorm",
        }
        for key, value in weights.items():
            if key.startswith("model.llm.layers."):
                layer, sub = key[len("model.llm.layers.") :].split(".", 1)
                base = f"language_model.model.layers.{layer}."
                if sub.startswith("mlp.experts."):
                    experts.setdefault(layer, {})[sub[len("mlp.experts.") :]] = value
                    continue
                if sub.startswith("attn."):
                    sub = sub[len("attn.") :]
                    name, _, leaf = sub.partition(".")
                    sub = attn_names.get(name, name) + ("." + leaf if leaf else "")
                    sub = {
                        "q_l2_clip.scale": "q_scale",
                        "rel_l2_clip.scale": "rel_scale",
                        "rel_logits_proj.proj": "rel_proj",
                    }.get(sub, sub)
                    for conv in ("k_sconv", "v_sconv"):
                        if sub == conv + ".weight":
                            sub = conv + ".conv.weight"
                            value = _to_mlx_conv(value)
                        elif sub == conv + ".static_w":
                            sub = conv + ".static_weight"
                        elif sub == conv + ".weight_proj":
                            sub += ".weight"
                    sub = "self_attn." + sub
                elif sub in ("attn_sconv.weight", "mlp_sconv.weight"):
                    sub = sub.replace(".weight", ".conv.weight")
                    value = _to_mlx_conv(value)
                elif sub in ("mlp.gate.weight", "mlp.gate.bias"):
                    sub = (
                        "mlp.gate_weight"
                        if sub.endswith("weight")
                        else "mlp.e_score_correction_bias"
                    )
                else:
                    name, _, leaf = sub.partition(".")
                    sub = norms.get(name, name) + "." + leaf
                out[base + sub] = value
            elif key.startswith("model.llm.attn_res."):
                sub = key[len("model.llm.attn_res.") :]
                if sub == "output_query":
                    target = "final_input.query"
                else:
                    kind, index = sub.split(".")
                    layer = int(index) + (kind == "attn_queries")
                    target = f"layers.{layer}.{'attention_input' if kind == 'attn_queries' else 'mlp_input'}.query"
                out["language_model.model." + target] = value
            elif key.startswith("model.llm."):
                sub = key[len("model.llm.") :]
                target = (
                    "language_model.lm_head.weight"
                    if sub == "unembed.weight"
                    else "language_model.model."
                    + sub.replace("embed.weight", "embed_tokens.weight")
                )
                out[target] = value
            elif key.startswith("model.visual."):
                sub = key[len("model.visual.") :].removeprefix("backbone.")
                sub = (
                    sub.replace("ln_pre.", "pre_layernorm.")
                    .replace(".ln_1.", ".layernorm_before.")
                    .replace(".ln_2.", ".layernorm_after.")
                )
                sub = (
                    sub.replace(".ls_1.gamma", ".lambda_1")
                    .replace(".ls_2.gamma", ".lambda_2")
                    .replace(".mlp.c_fc.", ".mlp.fc1.")
                    .replace(".mlp.c_proj.", ".mlp.fc2.")
                )
                out["vision_tower." + sub] = value
            elif key.startswith(("model.mtp.", "mtp.", "mtp_drafter.")):
                continue
            else:
                if key.startswith(("model.", "lm_head.")):
                    key = "language_model." + key
                prefix = "language_model.model.layers."
                if key.startswith(prefix):
                    layer, _, suffix = key[len(prefix):].partition(".")
                    key = f"{prefix}{layer}.{_MLX_EXPORT_LAYER_KEYS.get(suffix, suffix)}"
                key = _MLX_EXPORT_VISION_KEYS.get(key, key)
                out[key] = value
        for layer, tensors in experts.items():
            base = f"language_model.model.layers.{layer}.mlp.switch_mlp."
            for name, projections in (
                ("w13_weight", ("gate_proj", "up_proj")),
                ("w2_weight", ("down_proj",)),
            ):
                weight = tensors[name]
                packed = name + "_scale" in tensors
                values = {"weight": weight.view(mx.uint32) if packed else weight}
                if packed:
                    values["scales"] = tensors[name + "_scale"]
                if name + "_amax" in tensors:
                    values["weight_scale_2"] = (
                        tensors[name + "_amax"]
                        / tensors[name + "_outer_scale_denominator"]
                    )
                for suffix, v in values.items():
                    parts = (
                        _split_gate_up(v)
                        if len(projections) == 2 and suffix != "weight_scale_2"
                        else (v,) * len(projections)
                    )
                    for projection, part in zip(projections, parts):
                        out[base + projection + "." + suffix] = part
        base = "language_model.model."
        if base + "message_input.query" in out:
            if base + "message_queries" in out:
                raise ValueError("Conflicting Inkling loop message query tensors")
            out[base + "message_queries"] = out.pop(base + "message_input.query")
        if base + "middle_cycle_cross_pass_queries" in out:
            width = self.config.text_config.hidden_size
            attn_mlp = out.pop(base + "middle_cycle_cross_pass_queries").reshape(
                -1, 2, width
            )
            kv = out.pop(base + "middle_cycle_cross_pass_kv_queries").reshape(
                -1, 2, width
            )
            out[base + "message_queries"] = mx.concatenate(
                [attn_mlp[:, :1], kv, attn_mlp[:, 1:]], axis=1
            )
        return out
