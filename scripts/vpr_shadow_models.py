"""Pinned model adapters for the v1.4 offline VPR shadow diagnostic."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

try:
    from vpr_shadow_data import ShadowFrame, sha256
except ImportError:  # imported as scripts.vpr_shadow_models in unit tests
    from scripts.vpr_shadow_data import ShadowFrame, sha256


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_RETRIEVERS = (
    "mixvpr",
    "megaloc",
    "netvlad",
    "eigenplaces",
    "boq",
)


def load_and_validate_registry(
    project_root: Path, registry_path: Path
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if registry.get("schema") != "ascent_v1_4_vpr_shadow_model_registry_v1":
        raise ValueError("unexpected VPR model registry schema")
    records = list(registry["global_retrievers"].values()) + [
        registry["local_verifier"]
    ] + list(registry["design_references_only"].values())
    for record in records:
        repository = (project_root / record["repository_path"]).resolve()
        observed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed != record["repository_commit"]:
            raise ValueError(f"reference repository drift: {repository}")
    for record in registry["global_retrievers"].values():
        _check_file(project_root, record["weights_path"], record["weights_sha256"])
    verifier = registry["local_verifier"]
    _check_file(
        project_root,
        verifier["aliked_weights_path"],
        verifier["aliked_weights_sha256"],
    )
    _check_file(
        project_root,
        verifier["lightglue_weights_path"],
        verifier["lightglue_weights_sha256"],
    )
    return registry


def _check_file(project_root: Path, relative: str, expected: str) -> Path:
    path = (project_root / relative).resolve()
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"model asset hash mismatch: {path}")
    return path


def _rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read RGB keyframe {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class GlobalRetriever:
    def __init__(
        self,
        *,
        name: str,
        project_root: Path,
        registry: Mapping[str, Any],
        device: str,
    ) -> None:
        import torch

        self.name = str(name)
        self.project_root = Path(project_root).resolve()
        self.record = registry["global_retrievers"][self.name]
        self.device = torch.device(device)
        self.model = self._load_model().to(self.device).eval()

    def _load_model(self):
        import torch

        weights = (self.project_root / self.record["weights_path"]).resolve()
        repository = (
            self.project_root / self.record["repository_path"]
        ).resolve()
        if self.name == "mixvpr":
            sys.path.insert(0, str(repository))
            try:
                from models import helper
            finally:
                sys.path.pop(0)

            class InferenceMixVPR(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.backbone = helper.get_backbone(
                        "resnet50", False, 0, [4]
                    )
                    self.aggregator = helper.get_aggregator(
                        "MixVPR",
                        {
                            "in_channels": 1024,
                            "in_h": 20,
                            "in_w": 20,
                            "out_channels": 1024,
                            "mix_depth": 4,
                            "mlp_ratio": 1,
                            "out_rows": 4,
                        },
                    )

                def forward(self, value):
                    return self.aggregator(self.backbone(value))

            model = InferenceMixVPR()
            state = torch.load(weights, map_location="cpu")
            model.load_state_dict(state, strict=True)
            return model
        if self.name == "megaloc":
            spec = importlib.util.spec_from_file_location(
                "ascent_vpr_pinned_megaloc", repository / "megaloc_model.py"
            )
            if spec is None or spec.loader is None:
                raise ImportError("cannot load pinned MegaLoc module")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            from safetensors.torch import load_file

            model = module.MegaLoc()
            model.load_state_dict(load_file(str(weights)), strict=True)
            return model
        if self.name == "netvlad":
            torch_hub = (
                self.project_root
                / "artifacts/objectnav/vpr_shadow/model_cache/torch/hub"
            ).resolve()
            torch.hub.set_dir(str(torch_hub))
            hub_checkpoint = (
                torch_hub / self.record["torch_hub_relative_weights_path"]
            ).resolve()
            if not hub_checkpoint.is_file() or sha256(hub_checkpoint) != self.record[
                "weights_sha256"
            ]:
                raise ValueError("pinned NetVLAD torch-hub checkpoint mismatch")
            sys.path.insert(0, str(repository))
            try:
                from hloc.extractors.netvlad import NetVLAD
            finally:
                sys.path.pop(0)

            base = NetVLAD(
                {
                    "model_name": "VGG16-NetVLAD-Pitts30K",
                    "whiten": True,
                }
            )

            class InferenceNetVLAD(torch.nn.Module):
                def __init__(self, value) -> None:
                    super().__init__()
                    self.base = value

                def forward(self, value):
                    return self.base({"image": value})["global_descriptor"]

            return InferenceNetVLAD(base)
        if self.name == "eigenplaces":
            sys.path.insert(0, str(repository))
            try:
                from eigenplaces_model.layers import Flatten, GeM, L2Norm
            finally:
                sys.path.pop(0)
            import torchvision

            class InferenceEigenPlaces(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    backbone = torchvision.models.resnet50(weights=None)
                    self.backbone = torch.nn.Sequential(
                        *list(backbone.children())[:-2]
                    )
                    self.aggregation = torch.nn.Sequential(
                        L2Norm(),
                        GeM(),
                        Flatten(),
                        torch.nn.Linear(2048, 2048),
                        L2Norm(),
                    )

                def forward(self, value):
                    return self.aggregation(self.backbone(value))

            model = InferenceEigenPlaces()
            model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
            return model
        if self.name == "boq":
            source = repository / "src"
            sys.path.insert(0, str(source))
            try:
                from backbones import ResNet
                from boq import BoQ
            finally:
                sys.path.pop(0)

            class InferenceBoQ(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.backbone = ResNet(
                        backbone_name="resnet50",
                        pretrained=False,
                        unfreeze_n_blocks=0,
                        crop_last_block=True,
                    )
                    self.aggregator = BoQ(
                        in_channels=self.backbone.out_channels,
                        proj_channels=512,
                        num_queries=64,
                        num_layers=2,
                        row_dim=32,
                    )

                def forward(self, value):
                    descriptor, _ = self.aggregator(self.backbone(value))
                    return descriptor

            model = InferenceBoQ()
            model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
            return model
        raise ValueError(f"unknown retriever {self.name}")

    def encode(
        self, frames: Sequence[ShadowFrame], *, batch_size: int
    ) -> dict[str, np.ndarray]:
        import torch
        import torch.nn.functional as functional

        output: dict[str, np.ndarray] = {}
        height, width = map(int, self.record["input_size"])
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        effective_batch_size = min(
            int(batch_size), int(self.record.get("max_inference_batch_size", batch_size))
        )
        preprocessing = str(self.record.get("preprocessing", "imagenet"))
        if preprocessing not in {"imagenet", "unit_range"}:
            raise ValueError(f"unknown preprocessing contract {preprocessing}")
        with torch.inference_mode():
            for start in range(0, len(frames), effective_batch_size):
                subset = frames[start : start + effective_batch_size]
                tensors = []
                for frame in subset:
                    image = torch.from_numpy(
                        _rgb(frame.rgb_path).copy()
                    ).permute(2, 0, 1).float() / 255.0
                    if preprocessing == "imagenet":
                        image = (image - mean) / std
                    image = functional.interpolate(
                        image[None],
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )[0]
                    tensors.append(image)
                batch = torch.stack(tensors).to(self.device)
                descriptors = self.model(batch)
                descriptors = functional.normalize(descriptors, p=2, dim=1)
                values = descriptors.detach().cpu().numpy().astype(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError("non-finite global VPR descriptor")
                for frame, descriptor in zip(subset, values):
                    output[frame.frame_id] = descriptor
        return output


class LocalGeometryMatcher:
    def __init__(
        self,
        *,
        project_root: Path,
        registry: Mapping[str, Any],
        device: str,
    ) -> None:
        import torch

        self.project_root = Path(project_root).resolve()
        record = registry["local_verifier"]
        deps = (self.project_root / record["python_deps_path"]).resolve()
        repository = (self.project_root / record["repository_path"]).resolve()
        for path in (str(repository), str(deps)):
            if path not in sys.path:
                sys.path.insert(0, path)
        torch_hub = (
            self.project_root
            / "artifacts/objectnav/vpr_shadow/model_cache/torch/hub"
        ).resolve()
        torch.hub.set_dir(str(torch_hub))
        from lightglue import ALIKED, LightGlue

        self.device = torch.device(device)
        self.extractor = ALIKED(
            model_name="aliked-n16",
            max_num_keypoints=int(record["max_keypoints"]),
        ).eval().to(self.device)
        self.matcher = LightGlue(features="aliked").eval().to(self.device)
        self._features: dict[str, dict[str, Any]] = {}

    def features(self, frame: ShadowFrame) -> dict[str, Any]:
        import torch

        cached = self._features.get(frame.frame_id)
        if cached is not None:
            return cached
        image = torch.from_numpy(_rgb(frame.rgb_path).copy()).permute(2, 0, 1)
        image = image.float().to(self.device) / 255.0
        with torch.inference_mode():
            value = self.extractor.extract(image)
        value = _map_tensors(value, lambda tensor: tensor.detach().cpu())
        self._features[frame.frame_id] = value
        return value

    def match(
        self, source: ShadowFrame, target: ShadowFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch
        from lightglue.utils import rbd

        source_features = _map_tensors(
            self.features(source), lambda tensor: tensor.to(self.device)
        )
        target_features = _map_tensors(
            self.features(target), lambda tensor: tensor.to(self.device)
        )
        with torch.inference_mode():
            result = self.matcher(
                {"image0": source_features, "image1": target_features}
            )
        source_unbatched = rbd(source_features)
        target_unbatched = rbd(target_features)
        matches = rbd(result)["matches"]
        return (
            source_unbatched["keypoints"].detach().cpu().numpy(),
            target_unbatched["keypoints"].detach().cpu().numpy(),
            matches.detach().cpu().numpy(),
        )


def _map_tensors(value: Any, function):
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_map_tensors(item, function) for item in value)
    return value
