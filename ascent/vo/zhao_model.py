"""Minimal inference port of Zhao et al. 2021 PointNav-VO.

The network, running-statistics layer, resize/crop convention, hard depth
discretization, and top-down depth projection mirror the released PointNav-VO
implementation at dbff8719fe09cfb5a2dfbd730fd359b474fe3036.  The upstream
repository is Apache-2.0; portions of its visual encoder retain the original
Habitat MIT header.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FORWARD_CHECKPOINT_SHA256 = (
    "6b571bb717366f7d80f61e919b33a45ac2f45925201c4e3011b3239a2c42e586"
)
TURN_CHECKPOINT_SHA256 = (
    "c469643f9ab35c9e1058f31fbb672a5fa3adf582987a4388bdd020dd89faf1d9"
)
POINTNAV_VO_SOURCE_COMMIT = "dbff8719fe09cfb5a2dfbd730fd359b474fe3036"


class ZhaoCheckpointError(RuntimeError):
    """Raised when a released checkpoint cannot be verified or loaded."""


class Flatten(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.contiguous().view(value.size(0), -1).contiguous()


class RunningMeanAndVar(nn.Module):
    """Released PointNav-VO input normalization layer."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("_var", torch.zeros(1, channels, 1, 1))
        self.register_buffer("_count", torch.zeros(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("Zhao VO is inference-only in ASCENT")
        stdev = torch.sqrt(
            torch.maximum(self._var, torch.full_like(self._var, 1e-2))
        )
        return (value - self._mean) / stdev


def _conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        groups=groups,
    )


def _conv1x1(
    in_planes: int, out_planes: int, stride: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        ngroups: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            _conv3x3(inplanes, planes, stride),
            nn.GroupNorm(ngroups, planes),
            nn.ReLU(True),
            _conv3x3(planes, planes),
            nn.GroupNorm(ngroups, planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        output = self.convs(value)
        if self.downsample is not None:
            residual = self.downsample(value)
        return self.relu(output + residual)


class ResNet18(nn.Module):
    """The released 32-baseplane GroupNorm ResNet-18."""

    final_spatial_compress = 1.0 / 32.0

    def __init__(
        self, in_channels: int = 30, base_planes: int = 32, ngroups: int = 16
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(ngroups, base_planes),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = base_planes
        self.layer1 = self._make_layer(base_planes, 2, ngroups)
        self.layer2 = self._make_layer(
            base_planes * 2, 2, ngroups, stride=2
        )
        self.layer3 = self._make_layer(
            base_planes * 4, 2, ngroups, stride=2
        )
        self.layer4 = self._make_layer(
            base_planes * 8, 2, ngroups, stride=2
        )
        self.final_channels = self.inplanes

    def _make_layer(
        self, planes: int, blocks: int, ngroups: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes, stride),
                nn.GroupNorm(ngroups, planes),
            )
        layers = [
            BasicBlock(
                self.inplanes,
                planes,
                ngroups,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, ngroups))
        return nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv1(value)
        value = self.maxpool(value)
        value = self.layer1(value)
        value = self.layer2(value)
        value = self.layer3(value)
        return self.layer4(value)


class ZhaoVisualEncoder(nn.Module):
    def __init__(self, width: int = 341, height: int = 192) -> None:
        super().__init__()
        self._n_input_rgb = 6
        self._n_input_depth = 2
        self._n_input_discretized_depth = 20
        self._n_input_top_down_view = 2
        input_channels = 30

        self.running_mean_and_var = RunningMeanAndVar(input_channels)
        self.backbone = ResNet18(
            in_channels=input_channels, base_planes=32, ngroups=16
        )
        final_width = int(
            np.ceil(width * self.backbone.final_spatial_compress)
        )
        final_height = int(
            np.ceil(height * self.backbone.final_spatial_compress)
        )
        compression_channels = int(
            round(2048 / (final_width * final_height))
        )
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, compression_channels),
            nn.ReLU(True),
        )
        self.output_shape = (
            compression_channels,
            final_height,
            final_width,
        )

    @staticmethod
    def _split_pair(
        value: torch.Tensor, pair_channels: int
    ) -> list[torch.Tensor]:
        value = value.permute(0, 3, 1, 2)
        return [
            value[:, : pair_channels // 2],
            value[:, pair_channels // 2 :],
        ]

    def forward(
        self, observation_pairs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        rgb = observation_pairs["rgb"].permute(0, 3, 1, 2) / 255.0
        modalities = [
            [
                rgb[:, : self._n_input_rgb // 2],
                rgb[:, self._n_input_rgb // 2 :],
            ],
            self._split_pair(
                observation_pairs["depth"], self._n_input_depth
            ),
            self._split_pair(
                observation_pairs["discretized_depth"],
                self._n_input_discretized_depth,
            ),
            self._split_pair(
                observation_pairs["top_down_view"],
                self._n_input_top_down_view,
            ),
        ]
        # Released order: all previous-frame modalities, then all current ones.
        ordered = [item for pair in zip(*modalities) for item in pair]
        value = torch.cat(ordered, dim=1)
        value = self.running_mean_and_var(value)
        value = self.backbone(value)
        return self.compression(value)


class ZhaoVisualOdometryCNN(nn.Module):
    """Released vo_cnn_rgb_d_dd_top_down architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.visual_encoder = ZhaoVisualEncoder()
        flattened = int(np.prod(self.visual_encoder.output_shape))
        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Dropout(0.2),
            nn.Linear(flattened, 512),
            nn.ReLU(True),
        )
        self.output_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 3),
        )

    def forward(
        self, observation_pairs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.output_head(
            self.visual_fc(self.visual_encoder(observation_pairs))
        )


class NormalizedDepthTopDown:
    """Released top-down projection, including its numeric HFOV convention."""

    def __init__(
        self,
        *,
        min_depth: float,
        max_depth: float,
        height: int,
        width: int,
        hfov_numeric: float,
        kernel_size: int = 3,
        rows_around_center: int = 50,
    ) -> None:
        self._epsilon = 0.01
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._height = height
        self._width = width
        self._hfov_numeric = hfov_numeric
        self._kernel_size = kernel_size
        self._rows_around_center = rows_around_center
        focal = (width / 2.0) / np.tan(hfov_numeric / 2.0)
        self._intrinsic = torch.tensor(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

    def generate(self, normalized_depth: torch.Tensor) -> torch.Tensor:
        if normalized_depth.shape != (self._height, self._width, 1):
            raise ValueError(
                f"bad top-down depth shape {tuple(normalized_depth.shape)}"
            )
        depth_2d = normalized_depth[..., 0]
        nonzero = depth_2d != 0
        if not torch.any(nonzero):
            return torch.zeros_like(normalized_depth)
        nonzero_rows = torch.where(torch.any(nonzero, dim=1))[0]
        nonzero_cols = torch.where(torch.any(nonzero, dim=0))[0]
        min_row, max_row = int(nonzero_rows[0]), int(nonzero_rows[-1])
        min_col, max_col = int(nonzero_cols[0]), int(nonzero_cols[-1])
        cropped = depth_2d[
            min_row : max_row + 1, min_col : max_col + 1
        ]
        blurred_np = cv2.GaussianBlur(
            cropped.detach().cpu().numpy(),
            (self._kernel_size, self._kernel_size),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_ISOLATED,
        )
        blurred = torch.as_tensor(
            blurred_np, dtype=torch.float32, device=normalized_depth.device
        )
        coords = self._compute_coords(blurred, min_col)
        pixels = self._compute_pixels(coords[:2])
        unique_pixels, counts = torch.unique(
            pixels, dim=1, sorted=False, return_counts=True
        )
        valid = (
            (unique_pixels[0] >= 0)
            & (unique_pixels[0] < self._height)
            & (unique_pixels[1] >= 0)
            & (unique_pixels[1] < self._width)
        )
        output = torch.zeros(
            (self._height, self._width),
            dtype=torch.float32,
            device=normalized_depth.device,
        )
        output[
            unique_pixels[0, valid], unique_pixels[1, valid]
        ] = counts.float()[valid]
        bound = torch.max(output)
        if bound > 0:
            output = torch.clamp(output / bound, max=1.0)
        return output.unsqueeze(-1)

    def _compute_coords(
        self, depth: torch.Tensor, min_nonzero_col: int
    ) -> torch.Tensor:
        min_row = max(
            0, int(np.ceil(depth.shape[0] / 2)) - self._rows_around_center
        )
        max_row = min(
            depth.shape[0],
            int(np.ceil(depth.shape[0] / 2))
            + self._rows_around_center,
        )
        valid_rows = max_row - min_row
        v_coords, u_coords = torch.meshgrid(
            torch.arange(valid_rows, device=depth.device),
            torch.arange(depth.shape[1], device=depth.device),
            indexing="ij",
        )
        v_flat = v_coords.reshape(-1).float() + 0.5
        u_flat = (
            u_coords.reshape(-1).float() + float(min_nonzero_col) + 0.5
        )
        homogeneous = torch.stack(
            [u_flat, v_flat, torch.ones_like(u_flat)], dim=0
        )
        intrinsic_inv = torch.linalg.inv(
            self._intrinsic.to(depth.device)
        )
        coords = intrinsic_inv @ homogeneous
        true_depth = (
            depth[min_row:max_row].reshape(-1)
            * (self._max_depth - self._min_depth)
            + self._min_depth
        )
        coords = coords * true_depth
        return coords[[0, 2, 1]]

    def _compute_pixels(self, coords: torch.Tensor) -> torch.Tensor:
        intrinsic_inv = torch.linalg.inv(self._intrinsic.to(coords.device))
        rightmost = torch.tensor(
            [self._width - 0.5, 0.0, 1.0],
            dtype=torch.float32,
            device=coords.device,
        )
        max_x = (intrinsic_inv @ rightmost * self._max_depth)[0]
        min_x = -max_x
        normalized = coords.clone()
        normalized[0] = (normalized[0] - min_x) / (
            (max_x - min_x) * (1.0 + self._epsilon)
        )
        normalized[1] = (normalized[1] - self._min_depth) / (
            (self._max_depth - self._min_depth) * (1.0 + self._epsilon)
        )
        pixels = normalized[[1, 0]]
        pixels[0] = self._height - torch.ceil(
            self._height * pixels[0]
        )
        pixels[1] = torch.floor(self._width * pixels[1])
        return pixels.long()


@dataclass(frozen=True)
class PreparedZhaoFrame:
    rgb: torch.Tensor
    depth: torch.Tensor
    discretized_depth: torch.Tensor
    top_down_view: torch.Tensor


class ZhaoFramePreprocessor:
    def __init__(
        self,
        *,
        device: torch.device,
        source_min_depth: float,
        source_max_depth: float,
        source_hfov_degrees: float,
        target_width: int = 341,
        target_height: int = 192,
        checkpoint_hfov_degrees: float = 70.0,
        checkpoint_hfov_numeric: float = 70.0,
        checkpoint_min_depth: float = 0.1,
        checkpoint_max_depth: float = 10.0,
    ) -> None:
        self.device = device
        self.source_min_depth = source_min_depth
        self.source_max_depth = source_max_depth
        self.source_hfov_degrees = source_hfov_degrees
        self.target_width = target_width
        self.target_height = target_height
        self.checkpoint_hfov_degrees = checkpoint_hfov_degrees
        self.checkpoint_min_depth = checkpoint_min_depth
        self.checkpoint_max_depth = checkpoint_max_depth
        if not (
            0.0 < checkpoint_hfov_degrees <= source_hfov_degrees < 180.0
        ):
            raise ValueError(
                "checkpoint HFOV must be positive and no wider than source HFOV"
            )
        self.top_down = NormalizedDepthTopDown(
            min_depth=checkpoint_min_depth,
            max_depth=checkpoint_max_depth,
            height=target_height,
            width=target_width,
            # Preserve the released checkpoint's numeric convention exactly.
            # The upstream variable is named hfov_rad but receives YAML value
            # 70 without a degree-to-radian conversion during training.
            hfov_numeric=checkpoint_hfov_numeric,
        )

    def prepare(
        self, rgb: np.ndarray, normalized_depth: np.ndarray
    ) -> PreparedZhaoFrame:
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"bad VO RGB shape {rgb.shape}")
        if normalized_depth.ndim == 2:
            normalized_depth = normalized_depth[..., None]
        if (
            normalized_depth.ndim != 3
            or normalized_depth.shape[-1] != 1
            or normalized_depth.shape[:2] != rgb.shape[:2]
        ):
            raise ValueError(
                f"bad VO depth shape {normalized_depth.shape} for RGB {rgb.shape}"
            )
        if not np.isfinite(normalized_depth).all():
            raise ValueError("VO depth contains NaN or Inf")
        if normalized_depth.min() < -1e-5 or normalized_depth.max() > 1.00001:
            raise ValueError("VO normalized depth is outside [0,1]")

        rgb_tensor = torch.as_tensor(
            np.ascontiguousarray(rgb),
            dtype=torch.float32,
            device=self.device,
        )
        source_depth = torch.as_tensor(
            np.ascontiguousarray(normalized_depth),
            dtype=torch.float32,
            device=self.device,
        )
        metric_depth = source_depth * (
            self.source_max_depth - self.source_min_depth
        ) + self.source_min_depth
        checkpoint_depth = torch.clamp(
            (metric_depth - self.checkpoint_min_depth)
            / (self.checkpoint_max_depth - self.checkpoint_min_depth),
            min=0.0,
            max=1.0,
        )

        joined = torch.cat([rgb_tensor, checkpoint_depth], dim=-1)
        joined = self._calibrated_center_crop_resize(joined)
        rgb_out = joined[..., :3]
        depth_out = joined[..., 3:4]

        bins = torch.clamp(
            torch.floor(depth_out[..., 0] * 10).long(), min=0, max=9
        )
        discretized = F.one_hot(bins, num_classes=10).to(
            dtype=torch.float32
        )
        top_down = self.top_down.generate(depth_out)
        return PreparedZhaoFrame(
            rgb=rgb_out,
            depth=depth_out,
            discretized_depth=discretized,
            top_down_view=top_down,
        )

    def _calibrated_center_crop_resize(
        self, value: torch.Tensor
    ) -> torch.Tensor:
        """Match the checkpoint camera FOV/aspect without changing the sensor."""

        height, width = value.shape[:2]
        source_tangent = np.tan(
            np.radians(self.source_hfov_degrees) / 2.0
        )
        checkpoint_tangent = np.tan(
            np.radians(self.checkpoint_hfov_degrees) / 2.0
        )
        crop_width = int(
            round(width * checkpoint_tangent / source_tangent)
        )
        crop_height = int(
            round(
                crop_width
                * float(self.target_height)
                / float(self.target_width)
            )
        )
        if (
            crop_width <= 0
            or crop_height <= 0
            or crop_width > width
            or crop_height > height
        ):
            raise ValueError(
                "source camera cannot be center-cropped to checkpoint intrinsics: "
                f"source={(height, width)}, crop={(crop_height, crop_width)}"
            )
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2
        cropped = value[
            start_y : start_y + crop_height,
            start_x : start_x + crop_width,
        ]
        nchw = cropped.permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(
            nchw,
            size=(self.target_height, self.target_width),
            mode="area",
        )
        if resized.shape[-2:] != (self.target_height, self.target_width):
            raise ValueError(
                f"bad calibrated resize output {tuple(resized.shape)}"
            )
        return resized.squeeze(0).permute(1, 2, 0).contiguous()


def pair_prepared_frames(
    previous: PreparedZhaoFrame, current: PreparedZhaoFrame
) -> Dict[str, torch.Tensor]:
    return {
        "rgb": torch.cat([previous.rgb, current.rgb], dim=-1).unsqueeze(0),
        "depth": torch.cat(
            [previous.depth, current.depth], dim=-1
        ).unsqueeze(0),
        "discretized_depth": torch.cat(
            [previous.discretized_depth, current.discretized_depth], dim=-1
        ).unsqueeze(0),
        "top_down_view": torch.cat(
            [previous.top_down_view, current.top_down_view], dim=-1
        ).unsqueeze(0),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def _legacy_habitat_config_shim() -> Iterator[None]:
    """Temporarily provide the class name serialized by the 2021 checkpoint."""

    import habitat.config.default as habitat_config_default

    sentinel = object()
    original = getattr(habitat_config_default, "Config", sentinel)

    class Config(dict):
        def __getattr__(self, name: str):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value) -> None:
            self[name] = value

    Config.__module__ = habitat_config_default.__name__
    habitat_config_default.Config = Config
    try:
        yield
    finally:
        if original is sentinel:
            delattr(habitat_config_default, "Config")
        else:
            habitat_config_default.Config = original


def _load_checkpoint(path: Path, expected_sha256: str) -> dict:
    if not path.is_file():
        raise ZhaoCheckpointError(f"missing Zhao checkpoint: {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ZhaoCheckpointError(
            f"checkpoint SHA mismatch for {path}: {actual}"
        )
    with _legacy_habitat_config_shim():
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
    if not isinstance(checkpoint, dict) or "model_states" not in checkpoint:
        raise ZhaoCheckpointError(
            f"checkpoint lacks model_states: {path}"
        )
    return checkpoint


class ZhaoActionModels:
    """Strictly loaded official forward/left/right inference models."""

    def __init__(
        self, checkpoint_dir: Path, device: torch.device
    ) -> None:
        self.device = device
        checkpoint_dir = checkpoint_dir.resolve()
        forward_checkpoint = _load_checkpoint(
            checkpoint_dir / "act_forward.pth",
            FORWARD_CHECKPOINT_SHA256,
        )
        turn_checkpoint = _load_checkpoint(
            checkpoint_dir / "act_left_right_inv_joint.pth",
            TURN_CHECKPOINT_SHA256,
        )
        forward_states = forward_checkpoint["model_states"]
        turn_states = turn_checkpoint["model_states"]
        states = {
            1: forward_states.get(1, forward_states.get("1")),
            2: turn_states.get(2, turn_states.get("2")),
            3: turn_states.get(3, turn_states.get("3")),
        }
        if any(state is None for state in states.values()):
            raise ZhaoCheckpointError(
                "official checkpoints do not contain action states 1/2/3"
            )
        self.models: Dict[int, ZhaoVisualOdometryCNN] = {}
        for action, state in states.items():
            model = ZhaoVisualOdometryCNN()
            model.load_state_dict(state, strict=True)
            model.to(device)
            model.eval()
            self.models[action] = model
        del forward_checkpoint, turn_checkpoint, states

    @torch.inference_mode()
    def estimate(
        self,
        action: int,
        previous: PreparedZhaoFrame,
        current: PreparedZhaoFrame,
    ) -> np.ndarray:
        if action not in self.models:
            raise ValueError(f"Zhao has no motion model for action {action}")
        output = self.models[action](
            pair_prepared_frames(previous, current)
        )
        if output.shape != (1, 3):
            raise RuntimeError(
                f"bad Zhao output shape {tuple(output.shape)}"
            )
        result = output[0].detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(result).all():
            raise RuntimeError(f"non-finite Zhao output: {result.tolist()}")
        return result
