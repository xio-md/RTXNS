from __future__ import annotations

import hashlib
import os
import struct
import time as _time
import weakref
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from rtxns_genesis_style import CameraDesc as _BackendCameraDesc
from rtxns_genesis_style import EmbeddedTextureDesc as _BackendTextureDesc
from rtxns_genesis_style import GenesisStyleRenderer
from rtxns_genesis_style import SensorFrame as _BackendSensorFrame
from rtxns_genesis_style import SurfaceDesc as _BackendSurfaceDesc

from .errors import (
    InvalidStateError,
    RuntimeNotInitializedError,
    SceneDestroyedError,
    SceneNotInitializedError,
    UnsupportedFeatureError,
)
from .objects import (
    Camera,
    ColorTexture,
    DeformableShape,
    DisneySurface,
    Environment,
    GlassSurface,
    ImageTexture,
    Light,
    MatrixTransform,
    MetalSurface,
    ParticlesShape,
    PinholeCamera,
    PlasticSurface,
    Render,
    RigidShape,
    Shape,
    Surface,
    Texture,
    ThinLensCamera,
    UniformSubsurface,
    _OwnedObject,
    _OwnershipState,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_module_dir() -> Path:
    platform_dir = "windows-x64" if os.name == "nt" else "linux-x64"
    return _repo_root() / "bin" / platform_dir


@dataclass(frozen=True)
class RuntimeOptions:
    context_path: Path
    context_id: str
    backend: str
    device_index: int
    log_level: object
    enable_debug: bool
    enable_external_interop: bool
    module_dir: Path
    runtime_dir: Path


@dataclass(frozen=True)
class _UpdateOperation:
    name: str
    scope: str
    execution_layer: str
    status: str
    current_path: str
    next_target: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "scope": self.scope,
            "execution_layer": self.execution_layer,
            "status": self.status,
            "current_path": self.current_path,
            "next_target": self.next_target,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _SceneUpdatePlan:
    mode: str
    time: float
    dirty_categories: tuple[str, ...]
    dirty_sources: dict[str, tuple[str, ...]]
    force_render: bool
    backend_rebuilt: bool
    environment_applied: bool
    operations: tuple[_UpdateOperation, ...]
    blockers: tuple[str, ...]
    cxx_candidates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "time": float(self.time),
            "dirty_categories": tuple(self.dirty_categories),
            "dirty_sources": {
                category: tuple(sources)
                for category, sources in self.dirty_sources.items()
            },
            "force_render": bool(self.force_render),
            "backend_rebuilt": bool(self.backend_rebuilt),
            "environment_applied": bool(self.environment_applied),
            "operations": [operation.to_dict() for operation in self.operations],
            "blockers": tuple(self.blockers),
            "cxx_candidates": tuple(self.cxx_candidates),
        }


def _copy_update_plan_dict(plan: dict[str, object]) -> dict[str, object]:
    return {
        "mode": str(plan.get("mode", "not_run")),
        "time": float(plan.get("time", 0.0)),
        "dirty_categories": tuple(plan.get("dirty_categories", ())),
        "dirty_sources": {
            str(category): tuple(sources)
            for category, sources in dict(plan.get("dirty_sources", {})).items()
        },
        "force_render": bool(plan.get("force_render", False)),
        "backend_rebuilt": bool(plan.get("backend_rebuilt", False)),
        "environment_applied": bool(plan.get("environment_applied", False)),
        "operations": [dict(operation) for operation in list(plan.get("operations", ()))],
        "blockers": tuple(plan.get("blockers", ())),
        "cxx_candidates": tuple(plan.get("cxx_candidates", ())),
    }


def _copy_update_report(report: dict[str, object]) -> dict[str, object]:
    copied = {
        "mode": str(report.get("mode", "not_run")),
        "dirty_categories": tuple(report.get("dirty_categories", ())),
        "dirty_sources": {
            str(category): tuple(sources)
            for category, sources in dict(report.get("dirty_sources", {})).items()
        },
        "duration_ms": float(report.get("duration_ms", 0.0)),
        "time": float(report.get("time", 0.0)),
        "backend_rebuilt": bool(report.get("backend_rebuilt", False)),
        "environment_applied": bool(report.get("environment_applied", False)),
    }
    copied["plan"] = _copy_update_plan_dict(dict(report.get("plan", {})))
    return copied


class _ModuleRuntime:
    def __init__(self) -> None:
        self.options: Optional[RuntimeOptions] = None
        self.scenes: weakref.WeakSet[Scene] = weakref.WeakSet()
        self.generation = 0
        self._next_scene_index = 0

    def ensure_initialized(self) -> RuntimeOptions:
        if self.options is None:
            raise RuntimeNotInitializedError("Call DonutRenderPy.init(...) before create_scene().")
        return self.options

    def register_scene(self, scene: "Scene") -> None:
        self.scenes.add(scene)

    def allocate_scene_token(self) -> str:
        self._next_scene_index += 1
        return f"scene-{self.generation}-{self._next_scene_index}"

    def reset(self) -> None:
        self.options = None
        self.scenes = weakref.WeakSet()
        self.generation += 1
        self._next_scene_index = 0


_RUNTIME = _ModuleRuntime()


@dataclass(frozen=True)
class _ResolvedTexturePayload:
    raw_pixels: Optional[np.ndarray]
    encoded_bytes: Optional[bytes]
    mime_type: Optional[str]
    has_alpha: bool


def _texture_has_payload(texture: Optional[Texture]) -> bool:
    if not isinstance(texture, ImageTexture):
        return False
    return bool(texture.file) or bool(texture.image_data)


def _infer_image_mime_type(*, encoding: Optional[str], suffix: str, role: str) -> str:
    normalized_encoding = "" if encoding is None else str(encoding).strip().lower()
    normalized_suffix = str(suffix).strip().lower()
    if normalized_encoding in ("png", "image/png") or normalized_suffix == ".png":
        return "image/png"
    if normalized_encoding in ("jpg", "jpeg", "image/jpeg") or normalized_suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    raise UnsupportedFeatureError(
        f"{role} only supports PNG/JPEG encoded inputs when passing file/image bytes directly."
    )


def _resolve_texture_file_path(texture: ImageTexture) -> Path:
    candidate = Path(texture.file)
    if candidate.is_absolute():
        return candidate
    repo_candidate = _repo_root() / candidate
    if repo_candidate.exists():
        return repo_candidate
    return candidate


def _normalize_raw_texture_pixels(texture: ImageTexture, data: bytes, role: str) -> np.ndarray:
    width = int(texture.width)
    height = int(texture.height)
    channels = int(texture.channel)
    if width <= 0 or height <= 0:
        raise UnsupportedFeatureError(
            f"{role} raw ImageTexture requires positive width and height."
        )
    if channels <= 0:
        pixel_count = width * height
        if pixel_count <= 0 or len(data) % pixel_count != 0:
            raise UnsupportedFeatureError(
                f"{role} raw ImageTexture must provide channel count or image_data sized as width*height*channels."
            )
        channels = len(data) // pixel_count
    if channels not in (1, 2, 3, 4):
        raise UnsupportedFeatureError(
            f"{role} raw ImageTexture only supports 1-4 channels, got {channels}."
        )
    expected_size = width * height * channels
    if len(data) != expected_size:
        raise UnsupportedFeatureError(
            f"{role} raw ImageTexture expects {expected_size} bytes, got {len(data)}."
        )
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, channels).copy()


def _resolve_texture_payload(texture: Optional[Texture], *, role: str) -> Optional[_ResolvedTexturePayload]:
    if not isinstance(texture, ImageTexture):
        return None
    if not _texture_has_payload(texture):
        return None

    image_data = bytes(texture.image_data)
    encoding = None if texture.encoding is None else str(texture.encoding).strip().lower()
    if image_data:
        if encoding in ("", "raw", "raw8", "r8", "rg8", "rgb8", "rgba8"):
            raw_pixels = _normalize_raw_texture_pixels(texture, image_data, role)
            channels = int(raw_pixels.shape[2])
            return _ResolvedTexturePayload(
                raw_pixels=raw_pixels,
                encoded_bytes=None,
                mime_type=None,
                has_alpha=channels in (2, 4),
            )
        mime_type = _infer_image_mime_type(encoding=encoding, suffix="", role=role)
        return _ResolvedTexturePayload(
            raw_pixels=None,
            encoded_bytes=image_data,
            mime_type=mime_type,
            has_alpha=int(texture.channel) in (2, 4),
        )

    if not texture.file:
        return None
    path = _resolve_texture_file_path(texture)
    if not path.is_file():
        raise UnsupportedFeatureError(f"{role} texture file was not found: {path}")
    encoded_bytes = path.read_bytes()
    mime_type = _infer_image_mime_type(encoding=encoding, suffix=path.suffix, role=role)
    return _ResolvedTexturePayload(
        raw_pixels=None,
        encoded_bytes=encoded_bytes,
        mime_type=mime_type,
        has_alpha=int(texture.channel) in (2, 4),
    )


def _texture_factor_default(
    texture: Optional[Texture],
    channels: int,
    default: tuple[float, ...],
    textured_default: Optional[tuple[float, ...]],
) -> np.ndarray:
    resolved_default = default if textured_default is None else textured_default
    if isinstance(texture, ImageTexture) and _texture_has_payload(texture):
        return np.asarray(resolved_default, dtype=np.float32)
    return np.asarray(default, dtype=np.float32)


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)


def _encode_png(pixels: np.ndarray) -> bytes:
    image = np.ascontiguousarray(pixels, dtype=np.uint8)
    if image.ndim != 3:
        raise ValueError("PNG encoder expects an array of shape (H, W, C).")
    height, width, channels = image.shape
    if channels not in (1, 2, 3, 4):
        raise ValueError("PNG encoder only supports 1-4 channels.")
    color_type = {
        1: 0,
        2: 4,
        3: 2,
        4: 6,
    }[channels]
    raw_scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw_scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _rename_backend_texture(texture: _ResolvedTexturePayload, name: str) -> _BackendTextureDesc:
    if texture.encoded_bytes is not None and texture.mime_type is not None:
        return _BackendTextureDesc(
            name=name,
            image_bytes=texture.encoded_bytes,
            mime_type=texture.mime_type,
            has_alpha=bool(texture.has_alpha),
        )
    if texture.raw_pixels is None:
        raise ValueError("Texture payload does not contain usable image data.")
    channels = int(texture.raw_pixels.shape[2])
    return _BackendTextureDesc(
        name=name,
        image_bytes=_encode_png(texture.raw_pixels),
        mime_type="image/png",
        has_alpha=channels in (2, 4),
    )


def _common_raw_texture_size(payloads: tuple[Optional[_ResolvedTexturePayload], ...], role: str) -> tuple[int, int]:
    size: Optional[tuple[int, int]] = None
    for payload in payloads:
        if payload is None or payload.raw_pixels is None:
            continue
        current = (int(payload.raw_pixels.shape[0]), int(payload.raw_pixels.shape[1]))
        if size is None:
            size = current
        elif size != current:
            raise UnsupportedFeatureError(
                f"{role} currently requires all raw ImageTexture inputs to share the same resolution."
            )
    if size is None:
        raise UnsupportedFeatureError(f"{role} requires at least one raw ImageTexture input.")
    return size


def _raw_payload_to_channels(
    payload: Optional[_ResolvedTexturePayload],
    *,
    size: tuple[int, int],
    channels: int,
    default: tuple[int, ...],
) -> np.ndarray:
    height, width = size
    if payload is None:
        fill = np.asarray(default[:channels], dtype=np.uint8).reshape(1, 1, channels)
        return np.broadcast_to(fill, (height, width, channels)).copy()
    if payload.raw_pixels is None:
        raise UnsupportedFeatureError("This texture path requires raw ImageTexture image_data.")

    pixels = np.ascontiguousarray(payload.raw_pixels, dtype=np.uint8)
    if (pixels.shape[0], pixels.shape[1]) != size:
        raise UnsupportedFeatureError("ImageTexture resolutions do not match.")

    source_channels = int(pixels.shape[2])
    if source_channels == channels:
        return pixels.copy()
    if channels == 1:
        return pixels[:, :, :1].copy()
    if source_channels == 1:
        return np.repeat(pixels, channels, axis=2)
    if source_channels < channels:
        fill = np.asarray(default[:channels], dtype=np.uint8).reshape(1, 1, channels)
        result = np.broadcast_to(fill, (height, width, channels)).copy()
        result[:, :, :source_channels] = pixels
        return result
    return pixels[:, :, :channels].copy()


def _build_base_color_texture(
    material_name: str,
    base_texture: Optional[Texture],
    opacity_texture: Optional[Texture],
) -> Optional[_BackendTextureDesc]:
    base_payload = _resolve_texture_payload(base_texture, role=f"{material_name}.base_color")
    opacity_payload = _resolve_texture_payload(opacity_texture, role=f"{material_name}.opacity")
    if base_payload is None and opacity_payload is None:
        return None

    texture_name = f"{material_name}_base_color"
    if base_payload is not None and base_payload.encoded_bytes is not None and opacity_payload is None:
        return _rename_backend_texture(base_payload, texture_name)
    if opacity_payload is not None and opacity_payload.encoded_bytes is not None:
        raise UnsupportedFeatureError(
            f"{material_name}.opacity currently requires raw ImageTexture image_data so it can be packed into baseColor alpha."
        )
    if base_payload is not None and base_payload.encoded_bytes is not None:
        raise UnsupportedFeatureError(
            f"{material_name}.base_color file/encoded ImageTexture cannot currently be combined with a separate opacity ImageTexture."
        )

    size = _common_raw_texture_size((base_payload, opacity_payload), f"{material_name}.base_color")
    rgba = _raw_payload_to_channels(base_payload, size=size, channels=4, default=(255, 255, 255, 255))
    if opacity_payload is not None:
        alpha = _raw_payload_to_channels(opacity_payload, size=size, channels=1, default=(255,))
        rgba[:, :, 3] = alpha[:, :, 0]
    return _BackendTextureDesc(
        name=texture_name,
        image_bytes=_encode_png(rgba),
        mime_type="image/png",
        has_alpha=True,
    )


def _build_metallic_roughness_texture(
    material_name: str,
    roughness_texture: Optional[Texture],
    metallic_texture: Optional[Texture],
) -> Optional[_BackendTextureDesc]:
    roughness_payload = _resolve_texture_payload(roughness_texture, role=f"{material_name}.roughness")
    metallic_payload = _resolve_texture_payload(metallic_texture, role=f"{material_name}.metallic")
    if roughness_payload is None and metallic_payload is None:
        return None
    if (roughness_payload is not None and roughness_payload.encoded_bytes is not None) or (
        metallic_payload is not None and metallic_payload.encoded_bytes is not None
    ):
        raise UnsupportedFeatureError(
            f"{material_name} roughness/metallic textures currently require raw ImageTexture image_data because glTF expects a packed metallicRoughnessTexture."
        )

    size = _common_raw_texture_size((roughness_payload, metallic_payload), f"{material_name}.metallic_roughness")
    packed = np.zeros((size[0], size[1], 4), dtype=np.uint8)
    packed[:, :, 1] = _raw_payload_to_channels(
        roughness_payload,
        size=size,
        channels=1,
        default=(255,),
    )[:, :, 0]
    packed[:, :, 2] = _raw_payload_to_channels(
        metallic_payload,
        size=size,
        channels=1,
        default=(255,),
    )[:, :, 0]
    packed[:, :, 3] = 255
    return _BackendTextureDesc(
        name=f"{material_name}_metallic_roughness",
        image_bytes=_encode_png(packed),
        mime_type="image/png",
        has_alpha=False,
    )


def _build_emissive_texture(material_name: str, emission_texture: Optional[Texture]) -> Optional[_BackendTextureDesc]:
    payload = _resolve_texture_payload(emission_texture, role=f"{material_name}.emissive")
    if payload is None:
        return None
    texture_name = f"{material_name}_emissive"
    if payload.encoded_bytes is not None:
        return _rename_backend_texture(payload, texture_name)

    size = _common_raw_texture_size((payload,), f"{material_name}.emissive")
    rgb = _raw_payload_to_channels(payload, size=size, channels=3, default=(255, 255, 255))
    return _BackendTextureDesc(
        name=texture_name,
        image_bytes=_encode_png(rgb),
        mime_type="image/png",
        has_alpha=False,
    )


def _reject_unsupported_surface_textures(surface: Surface) -> None:
    unsupported_fields: list[str] = []
    always_unsupported = ("normal_map",)
    for field_name in always_unsupported:
        value = getattr(surface, field_name, None)
        if isinstance(value, ImageTexture) and _texture_has_payload(value):
            unsupported_fields.append(field_name)

    if isinstance(surface, PlasticSurface):
        extra_fields = ("ks", "eta")
    elif isinstance(surface, DisneySurface):
        extra_fields = ("eta", "specular_tint", "specular_trans", "diffuse_trans")
    elif isinstance(surface, MetalSurface):
        extra_fields = ("eta",)
    elif isinstance(surface, GlassSurface):
        extra_fields = ("kt", "eta")
    else:
        extra_fields = tuple()

    for field_name in extra_fields:
        value = getattr(surface, field_name, None)
        if isinstance(value, ImageTexture) and _texture_has_payload(value):
            unsupported_fields.append(field_name)

    if unsupported_fields:
        fields = ", ".join(sorted(set(unsupported_fields)))
        raise UnsupportedFeatureError(
            f"{type(surface).__name__} currently does not execute ImageTexture fields: {fields}."
        )


def _texture_to_values(
    texture: Optional[Texture],
    channels: int,
    default: tuple[float, ...],
    *,
    textured_default: Optional[tuple[float, ...]] = None,
) -> np.ndarray:
    if texture is None:
        return np.asarray(default, dtype=np.float32)

    if isinstance(texture, ColorTexture):
        values = np.asarray(texture.color, dtype=np.float32)
    elif isinstance(texture, ImageTexture):
        values = np.asarray(
            _texture_factor_default(texture, channels, default, textured_default)
            if texture.scale is None
            else texture.scale,
            dtype=np.float32,
        )
    elif np.isscalar(texture):
        values = np.asarray([texture], dtype=np.float32)
    else:
        raise UnsupportedFeatureError(f"Unsupported texture object: {type(texture).__name__}")

    if values.size == 1 and channels > 1:
        values = np.repeat(values, channels)
    elif values.size < channels:
        padding = np.asarray(default, dtype=np.float32)
        values = np.concatenate([values, padding[values.size : channels]])
    else:
        values = values[:channels]
    return np.clip(values.astype(np.float32), 0.0, 1.0)


def _texture_to_scalar(
    texture: Optional[Texture],
    default: float,
    *,
    textured_default: Optional[float] = None,
) -> float:
    resolved = None if textured_default is None else (float(textured_default),)
    return float(_texture_to_values(texture, 1, (default,), textured_default=resolved)[0])


def _light_to_emissive(light: Optional[Light]) -> np.ndarray:
    if light is None:
        return np.zeros((3,), dtype=np.float32)
    return _texture_to_values(
        light.emission,
        3,
        (0.0, 0.0, 0.0),
        textured_default=(1.0, 1.0, 1.0),
    ) * float(light.intensity)


def _surface_to_backend_desc(surface: Optional[Surface], light: Optional[Light]) -> _BackendSurfaceDesc:
    base_texture: Optional[Texture] = None
    roughness_texture: Optional[Texture] = None
    metallic_texture: Optional[Texture] = None
    opacity_texture: Optional[Texture] = None
    if surface is None:
        base = (0.8, 0.8, 0.8, 1.0)
        roughness = 1.0
        metallic = 0.0
        double_sided = False
    elif isinstance(surface, PlasticSurface):
        _reject_unsupported_surface_textures(surface)
        base_texture = surface.kd
        opacity_texture = surface.opacity
        roughness_texture = surface.roughness
        base = _texture_to_values(surface.kd, 4, (0.8, 0.8, 0.8, 1.0), textured_default=(1.0, 1.0, 1.0, 1.0))
        base[3] = _texture_to_scalar(surface.opacity, base[3], textured_default=1.0)
        roughness = _texture_to_scalar(surface.roughness, 1.0, textured_default=1.0)
        metallic = 0.0
        double_sided = surface.double_sided
    elif isinstance(surface, MetalSurface):
        _reject_unsupported_surface_textures(surface)
        base_texture = surface.kd
        opacity_texture = surface.opacity
        roughness_texture = surface.roughness
        base = _texture_to_values(surface.kd, 4, (0.85, 0.85, 0.85, 1.0), textured_default=(1.0, 1.0, 1.0, 1.0))
        base[3] = _texture_to_scalar(surface.opacity, base[3], textured_default=1.0)
        roughness = _texture_to_scalar(surface.roughness, 0.2, textured_default=1.0)
        metallic = 1.0
        double_sided = surface.double_sided
    elif isinstance(surface, GlassSurface):
        _reject_unsupported_surface_textures(surface)
        base_texture = surface.ks
        opacity_texture = surface.opacity
        roughness_texture = surface.roughness
        base = _texture_to_values(surface.ks, 4, (0.9, 0.9, 0.95, 0.08), textured_default=(1.0, 1.0, 1.0, 1.0))
        base[3] = _texture_to_scalar(surface.opacity, base[3], textured_default=1.0)
        roughness = _texture_to_scalar(surface.roughness, 0.02, textured_default=1.0)
        metallic = 0.0
        double_sided = surface.double_sided
    else:
        _reject_unsupported_surface_textures(surface)
        base_texture = getattr(surface, "kd", None)
        opacity_texture = getattr(surface, "opacity", None)
        roughness_texture = getattr(surface, "roughness", None)
        metallic_texture = getattr(surface, "metallic", None)
        base = _texture_to_values(
            base_texture,
            4,
            (0.8, 0.8, 0.8, 1.0),
            textured_default=(1.0, 1.0, 1.0, 1.0),
        )
        base[3] = _texture_to_scalar(opacity_texture, base[3], textured_default=1.0)
        roughness = _texture_to_scalar(roughness_texture, 1.0, textured_default=1.0)
        metallic = _texture_to_scalar(metallic_texture, 0.0, textured_default=1.0)
        double_sided = bool(getattr(surface, "double_sided", False))

    emissive = _light_to_emissive(light)
    emissive_texture = None if light is None else light.emission
    return _BackendSurfaceDesc(
        base_color=tuple(float(v) for v in base),
        roughness=float(np.clip(roughness, 0.0, 1.0)),
        metallic=float(np.clip(metallic, 0.0, 1.0)),
        emissive=tuple(float(v) for v in emissive),
        double_sided=bool(double_sided),
        base_color_texture=_build_base_color_texture(
            surface.name if surface is not None else "default_surface",
            base_texture,
            opacity_texture,
        ),
        metallic_roughness_texture=_build_metallic_roughness_texture(
            surface.name if surface is not None else "default_surface",
            roughness_texture,
            metallic_texture,
        ),
        emissive_texture=_build_emissive_texture(
            surface.name if surface is not None else "default_surface",
            emissive_texture,
        ),
    )


def _resolve_transform_matrix(transform: Optional[MatrixTransform]) -> np.ndarray:
    if transform is None:
        return np.eye(4, dtype=np.float32)
    if not isinstance(transform, MatrixTransform):
        raise UnsupportedFeatureError(
            f"{type(transform).__name__} is not supported in the Week 2 adapter. Use MatrixTransform."
        )
    return transform.matrix


def _camera_to_backend_desc(camera: Camera) -> _BackendCameraDesc:
    if isinstance(camera, ThinLensCamera):
        raise UnsupportedFeatureError("ThinLensCamera is part of the Week 2 API draft but not executable yet.")
    if not isinstance(camera, PinholeCamera):
        raise UnsupportedFeatureError(f"Unsupported camera type: {type(camera).__name__}")

    pose = _resolve_transform_matrix(camera.pose)
    position = tuple(float(v) for v in pose[:3, 3])
    lookat = tuple(float(v) for v in pose[:3, 3] - pose[:3, 2])
    up = tuple(float(v) for v in pose[:3, 1])
    return _BackendCameraDesc(
        uid=camera.name,
        pos=position,
        lookat=lookat,
        up=up,
        res=camera.film.resolution,
        fov=float(camera.fov),
        near=0.1,
        far=1000.0,
        model="pinhole",
    )


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.view(np.uint8))
    return hasher.hexdigest()


def _transform_fingerprint(transform: object) -> object:
    if transform is None:
        return None
    if isinstance(transform, MatrixTransform):
        return ("MatrixTransform", _array_digest(transform.matrix))
    return (type(transform).__name__, repr(transform))


def _texture_fingerprint(texture: Optional[Texture]) -> object:
    if texture is None:
        return None
    if isinstance(texture, ColorTexture):
        return ("ColorTexture", tuple(float(v) for v in texture.color))
    if isinstance(texture, ImageTexture):
        image_bytes = bytes(texture.image_data)
        image_digest = hashlib.blake2b(image_bytes, digest_size=12).hexdigest()
        return (
            "ImageTexture",
            texture.file,
            texture.width,
            texture.height,
            texture.channel,
            None if texture.scale is None else tuple(float(v) for v in texture.scale),
            texture.encoding,
            image_digest,
        )
    return (type(texture).__name__, repr(texture))


def _light_fingerprint(light: Optional[Light]) -> object:
    if light is None:
        return None
    return (
        type(light).__name__,
        light.name,
        _texture_fingerprint(light.emission),
        float(light.intensity),
        bool(light.two_sided),
        float(light.beam_angle),
    )


def _subsurface_fingerprint(subsurface: Optional[UniformSubsurface]) -> object:
    if subsurface is None:
        return None
    return (
        type(subsurface).__name__,
        subsurface.name,
        _texture_fingerprint(subsurface.thickness),
    )


def _surface_fingerprint(surface: Optional[Surface]) -> object:
    if surface is None:
        return None

    return (
        type(surface).__name__,
        surface.name,
        bool(surface.double_sided),
        _texture_fingerprint(getattr(surface, "roughness", None)),
        _texture_fingerprint(getattr(surface, "opacity", None)),
        _texture_fingerprint(getattr(surface, "normal_map", None)),
        _texture_fingerprint(getattr(surface, "kd", None)),
        _texture_fingerprint(getattr(surface, "ks", None)),
        _texture_fingerprint(getattr(surface, "kt", None)),
        _texture_fingerprint(getattr(surface, "eta", None)),
        _texture_fingerprint(getattr(surface, "metallic", None)),
        _texture_fingerprint(getattr(surface, "specular_tint", None)),
        _texture_fingerprint(getattr(surface, "specular_trans", None)),
        _texture_fingerprint(getattr(surface, "diffuse_trans", None)),
    )


def _environment_fingerprint(environment: Optional[Environment]) -> object:
    if environment is None:
        return None
    return (
        type(environment).__name__,
        environment.name,
        _texture_fingerprint(environment.emission),
        _transform_fingerprint(environment.transform),
    )


def _camera_fingerprint(camera: Camera) -> object:
    return (
        type(camera).__name__,
        camera.name,
        _transform_fingerprint(camera.pose),
        tuple(int(v) for v in camera.film.resolution),
        float(camera.filter.radius),
        int(camera.spp),
        float(getattr(camera, "fov", 0.0)),
        float(getattr(camera, "aperture", 0.0)),
        float(getattr(camera, "focal_len", 0.0)),
        float(getattr(camera, "focus_dis", 0.0)),
    )


def _object_reference_fingerprint(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Surface):
        return ("SurfaceObject", _surface_fingerprint(value))
    if isinstance(value, Light):
        return ("LightObject", _light_fingerprint(value))
    if isinstance(value, UniformSubsurface):
        return ("SubsurfaceObject", _subsurface_fingerprint(value))
    return ("Reference", str(value))


def _shape_state(shape: Shape) -> dict[str, object]:
    binding = (
        _object_reference_fingerprint(shape.surface),
        _object_reference_fingerprint(shape.emission),
        _object_reference_fingerprint(shape.subsurface),
        float(shape.clamp_normal),
    )

    if isinstance(shape, RigidShape):
        return {
            "binding": binding,
            "geometry": (
                type(shape).__name__,
                shape.name,
                str(shape.obj_path),
                _array_digest(shape.vertices),
                _array_digest(shape.triangles),
                _array_digest(shape.normals),
                _array_digest(shape.uvs),
            ),
            "transform": _transform_fingerprint(shape.transform),
        }

    if isinstance(shape, DeformableShape):
        return {
            "binding": binding,
            "geometry": (
                type(shape).__name__,
                shape.name,
                _array_digest(shape.vertices),
                _array_digest(shape.triangles),
                _array_digest(shape.normals),
                _array_digest(shape.uvs),
            ),
            "transform": None,
        }

    if isinstance(shape, ParticlesShape):
        return {
            "binding": binding,
            "geometry": (
                type(shape).__name__,
                shape.name,
                int(shape.subdivision),
                _array_digest(shape.centers),
                _array_digest(shape.radii),
            ),
            "transform": None,
        }

    return {
        "binding": binding,
        "geometry": (type(shape).__name__, shape.name, repr(shape)),
        "transform": None,
    }


class Scene:
    def __init__(self, options: RuntimeOptions, runtime_generation: int, scene_token: str) -> None:
        self._options = options
        self._runtime_generation = int(runtime_generation)
        self._scene_token = str(scene_token)
        self._render: Optional[Render] = None
        self._backend: Optional[GenesisStyleRenderer] = None
        self._initialized = False
        self._destroyed = False
        self._time = 0.0

        self._environments: dict[str, Environment] = {}
        self._emissions: dict[str, Light] = {}
        self._subsurfaces: dict[str, UniformSubsurface] = {}
        self._surfaces: dict[str, Surface] = {}
        self._shapes: dict[str, Shape] = {}
        self._cameras: dict[str, tuple[Camera, bool]] = {}
        self._dirty_flags: set[str] = set()
        self._dirty_sources: dict[str, set[str]] = {}
        self._owned_objects: weakref.WeakSet[_OwnedObject] = weakref.WeakSet()
        self._environment_snapshots: dict[str, object] = {}
        self._emission_snapshots: dict[str, object] = {}
        self._subsurface_snapshots: dict[str, object] = {}
        self._surface_snapshots: dict[str, object] = {}
        self._shape_snapshots: dict[str, dict[str, object]] = {}
        self._camera_snapshots: dict[str, object] = {}
        initial_plan = _SceneUpdatePlan(
            mode="not_run",
            time=0.0,
            dirty_categories=(),
            dirty_sources={},
            force_render=False,
            backend_rebuilt=False,
            environment_applied=False,
            operations=(),
            blockers=(),
            cxx_candidates=(),
        ).to_dict()
        self._last_update_plan: dict[str, object] = _copy_update_plan_dict(initial_plan)
        self._last_update_report: dict[str, object] = {
            "mode": "not_run",
            "dirty_categories": (),
            "dirty_sources": {},
            "duration_ms": 0.0,
            "time": 0.0,
            "backend_rebuilt": False,
            "environment_applied": False,
            "plan": _copy_update_plan_dict(initial_plan),
        }
        self._update_history: list[dict[str, object]] = []

    def _check_alive(self) -> None:
        if self._destroyed:
            raise SceneDestroyedError("This Scene has been destroyed.")

    def _check_ready(self) -> None:
        self._check_alive()
        if not self._initialized:
            raise SceneNotInitializedError("Call Scene.init(render) before Scene.update_*() or render_frame().")

    def _object_name(self, obj: object) -> str:
        if hasattr(obj, "name"):
            return str(getattr(obj, "name"))
        if hasattr(obj, "file"):
            path = str(getattr(obj, "file"))
            return path if path else "<unnamed>"
        return "<unnamed>"

    def _describe_object(self, obj: object) -> str:
        return f"{type(obj).__name__} '{self._object_name(obj)}'"

    def _mark_dirty(self, category: str, source: str) -> None:
        self._dirty_flags.add(category)
        self._dirty_sources.setdefault(category, set()).add(source)

    def _dirty_source_snapshot(self) -> dict[str, tuple[str, ...]]:
        return {
            category: tuple(sorted(sources))
            for category, sources in sorted(self._dirty_sources.items())
        }

    def _clear_dirty_state(self) -> None:
        self._dirty_flags.clear()
        self._dirty_sources.clear()

    def _make_operation(
        self,
        *,
        name: str,
        scope: str,
        execution_layer: str,
        status: str,
        current_path: str,
        next_target: str,
        reason: str,
    ) -> _UpdateOperation:
        return _UpdateOperation(
            name=name,
            scope=scope,
            execution_layer=execution_layer,
            status=status,
            current_path=current_path,
            next_target=next_target,
            reason=reason,
        )

    def _dirty_camera_names(self, dirty_sources: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
        result: list[str] = []
        for source in dirty_sources.get("camera", ()):
            if source.startswith("camera:"):
                result.append(source.split(":", 1)[1])
        return tuple(result)

    def _dirty_transform_shape_names(self, dirty_sources: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
        result: list[str] = []
        for source in dirty_sources.get("transform", ()):
            if not source.startswith("shape:"):
                continue
            payload = source.split(":", 1)[1]
            if "." not in payload:
                continue
            shape_name, suffix = payload.rsplit(".", 1)
            if suffix == "transform":
                result.append(shape_name)
        return tuple(result)

    def _queue_incremental_backend_updates(self, plan: dict[str, object]) -> None:
        if self._backend is None:
            raise InvalidStateError("Scene backend is unavailable while queuing incremental updates.")

        dirty_sources = {
            str(category): tuple(sources)
            for category, sources in dict(plan.get("dirty_sources", {})).items()
        }
        for camera_name in self._dirty_camera_names(dirty_sources):
            if camera_name not in self._cameras:
                continue
            camera, _denoise = self._cameras[camera_name]
            self._backend.update_camera(_camera_to_backend_desc(camera))

        for shape_name in self._dirty_transform_shape_names(dirty_sources):
            shape = self._shapes.get(shape_name)
            if not isinstance(shape, RigidShape):
                continue
            self._backend.update_rigid(shape_name, _resolve_transform_matrix(shape.transform))

    def _build_update_plan(self, *, time_value: float) -> dict[str, object]:
        dirty_categories = tuple(sorted(self._dirty_flags))
        dirty_sources = self._dirty_source_snapshot()
        operations: list[_UpdateOperation] = []
        blockers: list[str] = []
        cxx_candidates: list[str] = []

        def add_candidate(candidate: str) -> None:
            if candidate not in cxx_candidates:
                cxx_candidates.append(candidate)

        mode = "no_change"
        force_render = False
        backend_rebuilt = False
        environment_applied = False

        if self._backend is None:
            mode = "full_rebuild"
            force_render = True
            backend_rebuilt = True
            blockers.append("backend scene is not created yet, so the first update must replay the full Scene state.")
            operations.append(
                self._make_operation(
                    name="bootstrap_backend",
                    scope="scene",
                    execution_layer="python_adapter",
                    status="current",
                    current_path="Scene._rebuild_backend()",
                    next_target="native Scene resource handles",
                    reason="The adapter has no live backend scene yet, so environment, camera, and shape state must be rebuilt.",
                )
            )
            operations.append(
                self._make_operation(
                    name="advance_scene",
                    scope="scene",
                    execution_layer="backend_wrapper",
                    status="current",
                    current_path="GenesisStyleRenderer.update_scene(force_render=True)",
                    next_target="native Scene.update_scene(...)",
                    reason="The initial build still enters the GLB-backed load_scene() path in the current backend.",
                )
            )
            return _SceneUpdatePlan(
                mode=mode,
                time=float(time_value),
                dirty_categories=dirty_categories,
                dirty_sources=dirty_sources,
                force_render=force_render,
                backend_rebuilt=backend_rebuilt,
                environment_applied=environment_applied,
                operations=tuple(operations),
                blockers=tuple(blockers),
                cxx_candidates=tuple(cxx_candidates),
            ).to_dict()

        if not dirty_categories:
            operations.append(
                self._make_operation(
                    name="advance_scene",
                    scope="scene",
                    execution_layer="backend_wrapper",
                    status="current",
                    current_path="GenesisStyleRenderer.update_scene(force_render=False)",
                    next_target="native Scene.update_scene(...)",
                    reason="No dirty-state is pending, so we only advance the existing backend scene.",
                )
            )
            return _SceneUpdatePlan(
                mode=mode,
                time=float(time_value),
                dirty_categories=dirty_categories,
                dirty_sources=dirty_sources,
                force_render=force_render,
                backend_rebuilt=backend_rebuilt,
                environment_applied=environment_applied,
                operations=tuple(operations),
                blockers=tuple(blockers),
                cxx_candidates=tuple(cxx_candidates),
            ).to_dict()

        rebuild_categories = tuple(category for category in dirty_categories if category not in {"camera", "environment", "transform"})
        if rebuild_categories:
            mode = "full_rebuild"
            force_render = True
            backend_rebuilt = True

            if "camera" in dirty_categories:
                operations.append(
                    self._make_operation(
                        name="apply_camera",
                        scope="camera",
                        execution_layer="backend_wrapper",
                        status="current",
                        current_path="Scene._queue_incremental_backend_updates() -> GenesisStyleRenderer.update_camera()",
                        next_target="HeadlessPbrScene.set_camera(...)",
                        reason="Camera state can now be pushed through the backend without forcing a Scene rebuild.",
                    )
                )

            if "surface" in rebuild_categories:
                add_candidate("update_surface")
                blockers.append("surface dirty-state still falls back to full rebuild because material handles only exist in the GLB snapshot.")
                operations.append(
                    self._make_operation(
                        name="surface_incremental_candidate",
                        scope="surface.binding",
                        execution_layer="cxx_boundary",
                        status="planned_native",
                        current_path="full Scene rebuild via Scene._rebuild_backend()",
                        next_target="native surface/material update",
                        reason="The adapter can classify material changes, but the native backend cannot patch them in place yet.",
                    )
                )

            if "geometry" in rebuild_categories:
                add_candidate("update_shape")
                blockers.append("geometry dirty-state still falls back to full rebuild because mesh handles are not exposed incrementally.")
                operations.append(
                    self._make_operation(
                        name="geometry_incremental_candidate",
                        scope="shape.geometry",
                        execution_layer="cxx_boundary",
                        status="planned_native",
                        current_path="full Scene rebuild via Scene._rebuild_backend()",
                        next_target="native deformable/particles/mesh update",
                        reason="Shape add/update helpers already exist in Python, but the native scene only accepts a rebuilt GLB payload today.",
                    )
                )

            operations.append(
                self._make_operation(
                    name="rebuild_backend",
                    scope="scene",
                    execution_layer="python_adapter",
                    status="current",
                    current_path="Scene._rebuild_backend()",
                    next_target="incremental Scene.update_* handles",
                    reason=f"Dirty categories {', '.join(rebuild_categories)} still require a full backend replay in the current adapter.",
                )
            )
            operations.append(
                self._make_operation(
                    name="advance_scene",
                    scope="scene",
                    execution_layer="backend_wrapper",
                    status="current",
                    current_path="GenesisStyleRenderer.update_scene(force_render=True)",
                    next_target="native Scene.update_scene(...)",
                    reason="The rebuild path still refreshes the GLB snapshot before rendering can continue.",
                )
            )
            return _SceneUpdatePlan(
                mode=mode,
                time=float(time_value),
                dirty_categories=dirty_categories,
                dirty_sources=dirty_sources,
                force_render=force_render,
                backend_rebuilt=backend_rebuilt,
                environment_applied=environment_applied,
                operations=tuple(operations),
                blockers=tuple(blockers),
                cxx_candidates=tuple(cxx_candidates),
            ).to_dict()

        if "transform" in dirty_categories:
            mode = "incremental_camera_transform_environment"
        else:
            mode = "incremental_camera_environment"

        if "environment" in dirty_categories:
            environment_applied = True
            operations.append(
                self._make_operation(
                    name="apply_environment",
                    scope="environment",
                    execution_layer="backend_wrapper",
                    status="current",
                    current_path="Scene._apply_environment() -> GenesisStyleRenderer.set_ambient()/set_default_light()",
                    next_target="HeadlessPbrScene.set_ambient()/set_default_light()",
                    reason="Ambient lighting already updates the live backend scene without rebuilding geometry.",
                )
            )

        if "camera" in dirty_categories:
            operations.append(
                self._make_operation(
                    name="apply_camera",
                    scope="camera",
                    execution_layer="backend_wrapper",
                    status="current",
                    current_path="Scene._queue_incremental_backend_updates() -> GenesisStyleRenderer.update_camera()",
                    next_target="HeadlessPbrScene.set_camera(...)",
                    reason="Camera-only updates now stay on the live backend path instead of requiring a rebuild.",
                )
            )
        if "transform" in dirty_categories:
            operations.append(
                self._make_operation(
                    name="apply_rigid_transforms",
                    scope="shape.transform",
                    execution_layer="backend_wrapper",
                    status="current",
                    current_path="Scene._queue_incremental_backend_updates() -> GenesisStyleRenderer.update_rigid()",
                    next_target="HeadlessPbrScene.update_node_transform(...)",
                    reason="Rigid shape transforms now update the existing SceneGraph nodes without reloading GLB geometry.",
                )
            )

        operations.append(
            self._make_operation(
                name="advance_scene",
                scope="scene",
                execution_layer="backend_wrapper",
                status="current",
                current_path="GenesisStyleRenderer.update_scene(force_render=False)",
                next_target="native Scene.update_scene(...)",
                reason="Camera, transform, and environment changes can reuse the existing backend scene without a full reload.",
            )
        )
        return _SceneUpdatePlan(
            mode=mode,
            time=float(time_value),
            dirty_categories=dirty_categories,
            dirty_sources=dirty_sources,
            force_render=force_render,
            backend_rebuilt=backend_rebuilt,
            environment_applied=environment_applied,
            operations=tuple(operations),
            blockers=tuple(blockers),
            cxx_candidates=tuple(cxx_candidates),
        ).to_dict()

    def _execute_update_plan(self, plan: dict[str, object]) -> None:
        if bool(plan.get("backend_rebuilt", False)):
            self._rebuild_backend()
        elif bool(plan.get("environment_applied", False)):
            self._apply_environment()

        if self._backend is None:
            raise InvalidStateError("Scene backend is unavailable after executing the update plan.")
        if not bool(plan.get("backend_rebuilt", False)):
            self._queue_incremental_backend_updates(plan)
        self._backend.update_scene(
            force_render=bool(plan.get("force_render", False)),
            time=self._time,
        )

    def _record_update_report(
        self,
        *,
        plan: dict[str, object],
        duration_ms: float,
    ) -> None:
        copied_plan = _copy_update_plan_dict(plan)
        report = {
            "mode": str(copied_plan["mode"]),
            "dirty_categories": tuple(copied_plan["dirty_categories"]),
            "dirty_sources": {
                category: tuple(sources)
                for category, sources in dict(copied_plan["dirty_sources"]).items()
            },
            "duration_ms": float(duration_ms),
            "time": float(self._time),
            "backend_rebuilt": bool(copied_plan["backend_rebuilt"]),
            "environment_applied": bool(copied_plan["environment_applied"]),
            "plan": copied_plan,
        }
        self._last_update_plan = _copy_update_plan_dict(copied_plan)
        self._last_update_report = _copy_update_report(report)
        self._update_history.append(_copy_update_report(report))
        if len(self._update_history) > 64:
            self._update_history.pop(0)

    def get_dirty_state(self) -> dict[str, object]:
        self._check_alive()
        return {
            "categories": tuple(sorted(self._dirty_flags)),
            "sources": self._dirty_source_snapshot(),
        }

    def get_update_stats(self) -> dict[str, object]:
        self._check_alive()
        return _copy_update_report(self._last_update_report)

    def get_update_history(self) -> list[dict[str, object]]:
        self._check_alive()
        return [_copy_update_report(report) for report in self._update_history]

    def get_last_update_plan(self) -> dict[str, object]:
        self._check_alive()
        return _copy_update_plan_dict(self._last_update_plan)

    def preview_update_plan(self, time: Optional[float] = None) -> dict[str, object]:
        self._check_ready()
        time_value = self._time if time is None else float(time)
        return self._build_update_plan(time_value=time_value)

    def _iter_owned_dependencies(self, obj: _OwnedObject):
        if isinstance(obj, Environment):
            if isinstance(obj.emission, _OwnedObject):
                yield obj.emission
            return

        if isinstance(obj, Light):
            if isinstance(obj.emission, _OwnedObject):
                yield obj.emission
            return

        if isinstance(obj, UniformSubsurface):
            if isinstance(obj.thickness, _OwnedObject):
                yield obj.thickness
            return

        if isinstance(obj, (PlasticSurface, DisneySurface, MetalSurface, GlassSurface)):
            texture_fields = (
                "roughness",
                "opacity",
                "normal_map",
                "kd",
                "ks",
                "kt",
                "eta",
                "metallic",
                "specular_tint",
                "specular_trans",
                "diffuse_trans",
            )
            for field_name in texture_fields:
                value = getattr(obj, field_name, None)
                if isinstance(value, _OwnedObject):
                    yield value
            return

        if isinstance(obj, Shape):
            for value in (obj.surface, obj.emission, obj.subsurface):
                if isinstance(value, _OwnedObject):
                    yield value
            return

    def _track_owned_object(self, obj: Optional[_OwnedObject], *, role: str) -> None:
        if obj is None:
            return

        state = obj._donut_owner_state
        if state is None:
            obj._donut_owner_state = _OwnershipState(
                runtime_generation=self._runtime_generation,
                scene_token=self._scene_token,
                owner_kind=role,
                owner_name=self._object_name(obj),
            )
        elif state.destroyed:
            raise InvalidStateError(
                f"{self._describe_object(obj)} belongs to a destroyed Scene and cannot be reused. "
                f"Create a new {type(obj).__name__} instance."
            )
        elif state.runtime_generation != self._runtime_generation:
            raise InvalidStateError(
                f"{self._describe_object(obj)} belongs to a previous DonutRenderPy runtime and cannot be reused "
                "after destroy(). Create a fresh object after re-initializing the module."
            )
        elif state.scene_token != self._scene_token:
            raise InvalidStateError(
                f"{self._describe_object(obj)} is already attached to another Scene and cannot be reused here."
            )

        self._owned_objects.add(obj)
        for dependency in self._iter_owned_dependencies(obj):
            self._track_owned_object(dependency, role=f"{role}-dependency")

    def _mark_owned_objects_destroyed(self) -> None:
        for obj in list(self._owned_objects):
            state = obj._donut_owner_state
            if state is not None:
                state.destroyed = True
        self._owned_objects.clear()

    def init(self, render: Render) -> None:
        self._check_alive()
        if self._initialized:
            raise InvalidStateError("Scene.init(render) can only be called once per Scene.")
        self._track_owned_object(render, role="render")
        self._render = render
        self._initialized = True

    def _stage_environment(self, environment: Environment) -> None:
        state = _environment_fingerprint(environment)
        previous = self._environment_snapshots.get(environment.name)
        self._environments[environment.name] = environment
        self._environment_snapshots[environment.name] = state
        if previous != state:
            self._mark_dirty("environment", f"environment:{environment.name}")

    def update_environment(self, environment: Environment) -> None:
        self._check_ready()
        self._track_owned_object(environment, role="environment")
        self._stage_environment(environment)

    def _stage_emission(self, light: Light) -> None:
        state = _light_fingerprint(light)
        previous = self._emission_snapshots.get(light.name)
        self._emissions[light.name] = light
        self._emission_snapshots[light.name] = state
        if previous != state:
            self._mark_dirty("surface", f"emission:{light.name}")

    def update_emission(self, light: Light) -> None:
        self._check_ready()
        self._track_owned_object(light, role="light")
        self._stage_emission(light)

    def _stage_subsurface(self, subsurface: UniformSubsurface) -> None:
        state = _subsurface_fingerprint(subsurface)
        previous = self._subsurface_snapshots.get(subsurface.name)
        self._subsurfaces[subsurface.name] = subsurface
        self._subsurface_snapshots[subsurface.name] = state
        if previous != state:
            self._mark_dirty("surface", f"subsurface:{subsurface.name}")

    def update_subsurface(self, subsurface: UniformSubsurface) -> None:
        self._check_ready()
        self._track_owned_object(subsurface, role="subsurface")
        self._stage_subsurface(subsurface)

    def _stage_surface(self, surface: Surface) -> None:
        state = _surface_fingerprint(surface)
        previous = self._surface_snapshots.get(surface.name)
        self._surfaces[surface.name] = surface
        self._surface_snapshots[surface.name] = state
        if previous != state:
            self._mark_dirty("surface", f"surface:{surface.name}")

    def update_surface(self, surface: Surface) -> None:
        self._check_ready()
        self._track_owned_object(surface, role="surface")
        self._stage_surface(surface)

    def _stage_shape(self, shape: Shape) -> None:
        state = _shape_state(shape)
        previous = self._shape_snapshots.get(shape.name)

        if isinstance(shape.surface, Surface):
            self._surfaces[shape.surface.name] = shape.surface
            self._surface_snapshots[shape.surface.name] = _surface_fingerprint(shape.surface)
        if isinstance(shape.emission, Light):
            self._emissions[shape.emission.name] = shape.emission
            self._emission_snapshots[shape.emission.name] = _light_fingerprint(shape.emission)
        if isinstance(shape.subsurface, UniformSubsurface):
            self._subsurfaces[shape.subsurface.name] = shape.subsurface
            self._subsurface_snapshots[shape.subsurface.name] = _subsurface_fingerprint(shape.subsurface)

        self._shapes[shape.name] = shape
        self._shape_snapshots[shape.name] = state

        if previous is None:
            self._mark_dirty("geometry", f"shape:{shape.name}.create")
            return

        if previous["geometry"] != state["geometry"]:
            self._mark_dirty("geometry", f"shape:{shape.name}.geometry")
        if previous["transform"] != state["transform"]:
            self._mark_dirty("transform", f"shape:{shape.name}.transform")
        if previous["binding"] != state["binding"]:
            self._mark_dirty("surface", f"shape:{shape.name}.binding")

    def update_shape(self, shape: Shape) -> None:
        self._check_ready()
        self._track_owned_object(shape, role="shape")
        self._stage_shape(shape)

    def _stage_camera(self, camera: Camera, denoise: bool) -> None:
        state = _camera_fingerprint(camera)
        previous = self._camera_snapshots.get(camera.name)
        self._cameras[camera.name] = (camera, bool(denoise))
        self._camera_snapshots[camera.name] = state
        if previous != state:
            self._mark_dirty("camera", f"camera:{camera.name}")

    def update_camera(self, camera: Camera, denoise: bool) -> None:
        self._check_ready()
        self._track_owned_object(camera, role="camera")
        self._stage_camera(camera, denoise)

    def update_scene(self, time: float) -> None:
        self._check_ready()
        self._time = float(time)
        plan = self._build_update_plan(time_value=self._time)
        started = _time.perf_counter()
        self._execute_update_plan(plan)
        duration_ms = (_time.perf_counter() - started) * 1000.0
        self._record_update_report(
            plan=plan,
            duration_ms=duration_ms,
        )
        self._clear_dirty_state()

    def render_frame(self, camera: Camera | str) -> bytes:
        self._check_ready()
        camera_object = self._resolve_camera(camera)
        self.update_scene(time=self._time)
        if self._backend is None:
            raise InvalidStateError("Scene backend is unavailable after update_scene().")
        rgba = self._backend.render_camera_rgba(
            _camera_to_backend_desc(camera_object),
            force_render=False,
            time=self._time,
        )
        return np.ascontiguousarray(rgba, dtype=np.uint8).tobytes()

    def render_sensor(
        self,
        camera: Camera | str,
        products: tuple[str, ...] = (
            "color",
            "depth",
            "normal",
            "instance",
            "semantic",
        ),
    ) -> _BackendSensorFrame:
        self._check_ready()
        camera_object = self._resolve_camera(camera)
        self.update_scene(time=self._time)
        if self._backend is None:
            raise InvalidStateError("Scene backend is unavailable after update_scene().")
        return self._backend.render_sensor(
            _camera_to_backend_desc(camera_object),
            products=products,
            force_render=False,
            time=self._time,
        )

    def render_sensor_batch(
        self,
        cameras: tuple[Camera | str, ...],
        products: tuple[str, ...] = (
            "color",
            "depth",
            "normal",
            "instance",
            "semantic",
        ),
    ) -> tuple[_BackendSensorFrame, ...]:
        self._check_ready()
        camera_objects = tuple(self._resolve_camera(camera) for camera in cameras)
        if not camera_objects:
            return ()
        self.update_scene(time=self._time)
        if self._backend is None:
            raise InvalidStateError("Scene backend is unavailable after update_scene().")
        return self._backend.render_sensor_batch(
            [_camera_to_backend_desc(camera) for camera in camera_objects],
            products=products,
            force_render=False,
            time=self._time,
        )

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        if self._backend is not None:
            self._backend.destroy()
            self._backend = None
        self._environments.clear()
        self._emissions.clear()
        self._subsurfaces.clear()
        self._surfaces.clear()
        self._shapes.clear()
        self._cameras.clear()
        self._environment_snapshots.clear()
        self._emission_snapshots.clear()
        self._subsurface_snapshots.clear()
        self._surface_snapshots.clear()
        self._shape_snapshots.clear()
        self._camera_snapshots.clear()
        self._clear_dirty_state()
        self._mark_owned_objects_destroyed()

    def _resolve_surface(self, surface_ref: Optional[Surface | str]) -> Optional[Surface]:
        if surface_ref is None:
            return None
        if isinstance(surface_ref, Surface):
            return surface_ref
        if surface_ref not in self._surfaces:
            raise InvalidStateError(f"Unknown surface reference '{surface_ref}'.")
        return self._surfaces[surface_ref]

    def _resolve_light(self, light_ref: Optional[Light | str]) -> Optional[Light]:
        if light_ref is None:
            return None
        if isinstance(light_ref, Light):
            return light_ref
        if light_ref not in self._emissions:
            raise InvalidStateError(f"Unknown emission reference '{light_ref}'.")
        return self._emissions[light_ref]

    def _resolve_camera(self, camera_ref: Camera | str) -> Camera:
        if isinstance(camera_ref, Camera):
            self._track_owned_object(camera_ref, role="camera")
            denoise = self._cameras.get(camera_ref.name, (camera_ref, False))[1]
            self._stage_camera(camera_ref, denoise)
            return camera_ref
        if camera_ref not in self._cameras:
            raise InvalidStateError(f"Unknown camera reference '{camera_ref}'.")
        return self._cameras[camera_ref][0]

    def _rebuild_backend(self) -> None:
        if self._backend is not None:
            self._backend.destroy()

        self._backend = GenesisStyleRenderer(
            module_dir=self._options.module_dir,
            runtime_dir=self._options.runtime_dir,
            backend=self._options.backend,
            device_index=self._options.device_index,
            enable_debug=self._options.enable_debug,
            enable_external_interop=self._options.enable_external_interop,
        )
        self._apply_environment()

        for camera, _denoise in self._cameras.values():
            self._backend.add_camera(_camera_to_backend_desc(camera))

        for shape in self._shapes.values():
            self._sync_shape(shape)

    def _apply_environment(self) -> None:
        if not self._environments:
            return

        latest = next(reversed(self._environments.values()))
        if latest.emission is None:
            return

        rgb = tuple(float(v) for v in _texture_to_values(latest.emission, 3, (0.03, 0.04, 0.06)))
        self._backend.set_ambient(rgb, rgb)
        # The native backend rejects non-positive irradiance, so disable the fallback
        # directional light by zeroing its color instead of passing irradiance=0.
        self._backend.set_default_light(direction=(-0.4, -1.0, -0.6), color=(0.0, 0.0, 0.0), irradiance=1.0)

    def _sync_shape(self, shape: Shape) -> None:
        surface = self._resolve_surface(shape.surface)
        light = self._resolve_light(shape.emission)
        backend_surface = _surface_to_backend_desc(surface, light)
        self._backend.add_surface(shape.name, backend_surface)

        if isinstance(shape, RigidShape):
            if shape.obj_path:
                raise UnsupportedFeatureError("RigidShape.obj_path is part of the draft API but not executed yet.")
            self._backend.add_rigid(shape.name, shape.vertices, shape.triangles, shape.normals, shape.uvs)
            if shape.transform is not None:
                self._backend.update_rigid(shape.name, _resolve_transform_matrix(shape.transform))
            return

        if isinstance(shape, DeformableShape):
            self._backend.add_deformable(shape.name)
            self._backend.update_deformable(shape.name, shape.vertices, shape.triangles, shape.normals, shape.uvs)
            return

        if isinstance(shape, ParticlesShape):
            radius = float(shape.radii[0]) if shape.radii.size > 0 else 0.05
            self._backend.add_particles(shape.name, radius=radius)
            self._backend.update_particles(shape.name, shape.centers, radius=radius, particles_radii=shape.radii)
            return

        raise UnsupportedFeatureError(f"Unsupported shape type: {type(shape).__name__}")


def init(
    context_path: str | os.PathLike[str] | None = None,
    context_id: str = "",
    backend: str = "vulkan",
    device_index: int = -1,
    log_level: object = None,
    *,
    enable_debug: bool = False,
    enable_external_interop: bool = False,
    module_dir: str | os.PathLike[str] | None = None,
    runtime_dir: str | os.PathLike[str] | None = None,
) -> None:
    resolved_module_dir = Path(module_dir) if module_dir is not None else _default_module_dir()
    if runtime_dir is not None:
        resolved_runtime_dir = Path(runtime_dir)
    elif context_path is not None:
        resolved_runtime_dir = Path(context_path)
    else:
        resolved_runtime_dir = _repo_root()

    options = RuntimeOptions(
        context_path=Path(context_path) if context_path is not None else resolved_runtime_dir,
        context_id=str(context_id),
        backend=str(backend),
        device_index=int(device_index),
        log_level=log_level,
        enable_debug=bool(enable_debug),
        enable_external_interop=bool(enable_external_interop),
        module_dir=resolved_module_dir,
        runtime_dir=resolved_runtime_dir,
    )
    if _RUNTIME.options is None:
        _RUNTIME.options = options
        return
    if _RUNTIME.options != options:
        raise InvalidStateError("DonutRenderPy is already initialized with different options.")


def create_scene() -> Scene:
    options = _RUNTIME.ensure_initialized()
    scene = Scene(options, runtime_generation=_RUNTIME.generation, scene_token=_RUNTIME.allocate_scene_token())
    _RUNTIME.register_scene(scene)
    return scene


def destroy() -> None:
    for scene in list(_RUNTIME.scenes):
        scene.destroy()
    _RUNTIME.reset()
