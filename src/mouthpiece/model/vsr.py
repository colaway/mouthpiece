"""The Auto-AVSR visual speech recognition model (Conformer encoder + transformer decoder + CTC).

Equivalent to ESPnet's `e2e_asr_transformer.E2E` as used by
mpc001/Visual_Speech_Recognition_for_Multiple_Languages (Apache License 2.0), reduced to
inference. The checkpoint is a plain state dict; its keys are kept unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from mouthpiece.model.transformer import ConformerEncoder, TransformerDecoder


class ModelLoadError(RuntimeError):
    """Checkpoint/config missing or not the architecture this port supports."""


@dataclass(frozen=True)
class VSRConfig:
    odim: int = 5049
    adim: int = 768
    aheads: int = 12
    eunits: int = 3072
    elayers: int = 12
    dunits: int = 3072
    dlayers: int = 6
    cnn_module_kernel: int = 31
    zero_triu: bool = False

    @classmethod
    def from_json(cls, path: str | Path, odim: int) -> VSRConfig:
        """Read ESPnet's model.json ([idim, odim, args]). `odim` comes from the weights:
        the json's own odim (41) is stale and does not match the unigram5000 head."""
        raw: Any = json.loads(Path(path).read_text())
        args: dict[str, Any] = raw if isinstance(raw, dict) else raw[2]
        expect = {
            "transformer_input_layer": "conv3d",
            "transformer_encoder_attn_layer_type": "rel_mha",
            "rel_pos_type": "latest",
        }
        for key, value in expect.items():
            if args.get(key) != value:
                raise ModelLoadError(f"unsupported model config: {key}={args.get(key)!r}")
        if not (args.get("macaron_style") and args.get("use_cnn_module")):
            raise ModelLoadError("only macaron-style Conformer encoders are supported")
        if args.get("relu_type", "swish") != "swish":
            raise ModelLoadError(f"unsupported relu_type {args.get('relu_type')!r}")
        return cls(
            odim=odim,
            adim=int(args["adim"]),
            aheads=int(args["aheads"]),
            eunits=int(args["eunits"]),
            elayers=int(args["elayers"]),
            dunits=int(args["dunits"]),
            dlayers=int(args["dlayers"]),
            cnn_module_kernel=int(args["cnn_module_kernel"]),
            zero_triu=bool(args.get("zero_triu", False)),
        )


class CTC(nn.Module):
    def __init__(self, odim: int, adim: int) -> None:
        super().__init__()
        self.ctc_lo = nn.Linear(adim, odim)

    def log_softmax(self, hs: Tensor) -> Tensor:
        return torch.log_softmax(self.ctc_lo(hs), dim=-1)


class VSRModel(nn.Module):
    """Input (B, 1, T, 88, 88) normalized grayscale -> encoder states (B, T, adim)."""

    blank = 0

    def __init__(self, cfg: VSRConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.odim = cfg.odim
        self.sos = self.eos = cfg.odim - 1
        self.encoder = ConformerEncoder(
            cfg.adim, cfg.aheads, cfg.eunits, cfg.elayers, cfg.cnn_module_kernel, cfg.zero_triu
        )
        self.decoder = TransformerDecoder(cfg.odim, cfg.adim, cfg.aheads, cfg.dunits, cfg.dlayers)
        self.ctc = CTC(cfg.odim, cfg.adim)

    @torch.no_grad()
    def encode(self, video: Tensor) -> Tensor:
        """(T, 88, 88) float clip -> (T, adim) encoder output on the model's device."""
        device = next(self.parameters()).device
        x = video.to(device=device, dtype=torch.float32)[None, None]
        out: Tensor = self.encoder(x)[0]
        return out


def load_state_dict(path: str | Path) -> dict[str, Tensor]:
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(f"checkpoint not found: {path} (run ./scripts/fetch_models.sh)")
    sd: Any = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    if not isinstance(sd, dict):
        raise ModelLoadError(f"{path} is not a state dict")
    return sd


def load_vsr_model(
    checkpoint: str | Path,
    model_config: str | Path,
    device: torch.device | str = "cpu",
    vocab_size: int | None = None,
) -> VSRModel:
    """Build the model from model.json and load weights (weights_only=True)."""
    sd = load_state_dict(checkpoint)
    if "ctc.ctc_lo.weight" not in sd:
        raise ModelLoadError("checkpoint has no CTC head (ctc.ctc_lo.weight)")
    odim = int(sd["ctc.ctc_lo.weight"].shape[0])
    if vocab_size is not None and vocab_size != odim:
        raise ModelLoadError(f"tokenizer has {vocab_size} tokens but the model outputs {odim}")
    model = VSRModel(VSRConfig.from_json(model_config, odim))
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()
