from __future__ import annotations

import importlib
import shutil
import struct
import sys
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .glb_builder import EmbeddedTextureDesc, GlbSceneBuilder
from .sensor import (
    SENSOR_PRODUCTS,
    SensorFrame,
    decode_sensor_frames,
    normalize_sensor_products,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_module_dir() -> Path:
    return _repo_root() / "bin" / "windows-x64"


def _default_runtime_dir() -> Path:
    return _repo_root()


def _import_native_renderer_module():
    errors: list[Exception] = []
    for module_name in ("DonutRenderPyNative", "RtxRenderPy"):
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(exc)
    details = "; ".join(str(exc) for exc in errors)
    raise ImportError(f"Failed to import Donut native renderer module. {details}")


def _shape_name(name: str, batch_index: Optional[int]) -> str:
    return name if batch_index is None else f"{name}_{batch_index}"


def _as_float_array(values: Any, length: Optional[int] = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if length is not None and array.size != length:
        raise ValueError(f"Expected {length} values, got {array.size}.")
    return array


def _coerce_vertices(vertices: Any) -> np.ndarray:
    array = np.asarray(vertices, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Vertices must have shape (N, 3).")
    return np.ascontiguousarray(array)


def _coerce_triangles(triangles: Any) -> np.ndarray:
    array = np.asarray(triangles, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Triangles must have shape (N, 3).")
    if array.size > 0 and np.min(array) < 0:
        raise ValueError("Triangle indices must be non-negative.")
    return np.ascontiguousarray(array.astype(np.uint32))


_PARTICLE_RADIUS_MIN = 1.0e-5
_PARTICLE_RADIUS_MAX = 1.0e3


def _validate_finite_xyz(array: np.ndarray, *, label: str) -> None:
    if array.size == 0:
        return
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values (NaN or Inf).")


def _validate_triangle_indices(triangles: np.ndarray, vertex_count: int) -> None:
    if triangles.size == 0 or vertex_count <= 0:
        return
    max_idx = int(np.max(triangles))
    if max_idx >= vertex_count:
        raise ValueError(f"Triangle index {max_idx} is out of range for {vertex_count} vertices.")


def _coerce_normals(normals: Any, vertex_count: int) -> np.ndarray:
    if normals is None:
        return np.empty((0, 3), dtype=np.float32)
    array = np.asarray(normals, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] != vertex_count:
        raise ValueError("Normals must have shape (N, 3) and match the vertex count.")
    return np.ascontiguousarray(array)


def _coerce_uvs(uvs: Any, vertex_count: int) -> np.ndarray:
    if uvs is None:
        return np.empty((0, 2), dtype=np.float32)
    array = np.asarray(uvs, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] != vertex_count:
        raise ValueError("UVs must have shape (N, 2) and match the vertex count.")
    return np.ascontiguousarray(array)


def _coerce_transform(matrix: Any) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.shape != (4, 4):
        raise ValueError("Transforms must have shape (4, 4).")
    return np.ascontiguousarray(array)


def _transform_to_gltf_node_matrix(matrix: Any) -> np.ndarray:
    return _coerce_transform(matrix)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1.0e-12
    result = np.zeros_like(vectors, dtype=np.float32)
    if np.any(valid):
        result[valid] = vectors[valid] / lengths[valid]
    if np.any(~valid):
        result[~valid] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return result


def _compute_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    if triangles.size == 0 or vertices.size == 0:
        return normals

    tri_vertices = vertices[triangles]
    face_normals = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0],
    )
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    return _normalize_vectors(normals)


def _transform_vertices(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    return np.ascontiguousarray(vertices @ linear.T + translation, dtype=np.float32)


def _transform_normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if normals.size == 0:
        return normals
    linear = matrix[:3, :3]
    normal_matrix = np.linalg.inv(linear).T
    return np.ascontiguousarray(_normalize_vectors(normals @ normal_matrix.T), dtype=np.float32)


def _extract_texture_value(
    texture: Any,
    channels: int,
    default: Sequence[float],
    *,
    textured_default: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if texture is None:
        return np.asarray(default, dtype=np.float32)

    if isinstance(texture, Mapping):
        if "image_data" in texture or "file" in texture:
            fallback = default if textured_default is None else textured_default
            return _texture_scale_values(texture, channels, fallback)
        if "textures" in texture:
            items = texture.get("textures") or []
            return _extract_texture_value(items[0] if items else None, channels, default, textured_default=textured_default)
        if "color" in texture:
            texture = texture["color"]
        elif "image_color" in texture:
            texture = texture["image_color"]

    if hasattr(texture, "textures"):
        items = getattr(texture, "textures")
        texture = items[0] if items else None
        return _extract_texture_value(texture, channels, default, textured_default=textured_default)

    if texture is None:
        return np.asarray(default, dtype=np.float32)

    if _texture_has_payload(texture):
        fallback = default if textured_default is None else textured_default
        return _texture_scale_values(texture, channels, fallback)

    if hasattr(texture, "color"):
        values = np.asarray(getattr(texture, "color"), dtype=np.float32).reshape(-1)
    elif hasattr(texture, "mean_color"):
        values = np.asarray(texture.mean_color(), dtype=np.float32).reshape(-1)
        image_color = getattr(texture, "image_color", None)
        if image_color is not None:
            scale = np.asarray(image_color, dtype=np.float32).reshape(-1)
            values = values * scale[: values.size]
    elif hasattr(texture, "image_color"):
        values = np.asarray(getattr(texture, "image_color"), dtype=np.float32).reshape(-1)
    elif np.isscalar(texture):
        values = np.asarray([texture], dtype=np.float32)
    else:
        values = np.asarray(texture, dtype=np.float32).reshape(-1)

    if values.size == 0:
        values = np.asarray(default, dtype=np.float32)
    if values.size == 1 and channels > 1:
        values = np.repeat(values, channels)
    elif values.size < channels:
        tail = np.asarray(default, dtype=np.float32)
        values = np.concatenate([values, tail[values.size : channels]])
    else:
        values = values[:channels]
    return np.clip(values.astype(np.float32), 0.0, 1.0)


def _texture_has_payload(texture: Any) -> bool:
    if texture is None:
        return False
    if isinstance(texture, Mapping):
        return bool(texture.get("file")) or bool(texture.get("image_data"))
    return bool(getattr(texture, "file", "")) or bool(getattr(texture, "image_data", b""))


def _texture_file(texture: Any) -> str:
    if isinstance(texture, Mapping):
        return str(texture.get("file", ""))
    return str(getattr(texture, "file", ""))


def _texture_image_data(texture: Any) -> bytes:
    if isinstance(texture, Mapping):
        value = texture.get("image_data", b"")
    else:
        value = getattr(texture, "image_data", b"")
    return value if isinstance(value, bytes) else str(value).encode("utf-8")


def _texture_width(texture: Any) -> int:
    if isinstance(texture, Mapping):
        return int(texture.get("width", 0))
    return int(getattr(texture, "width", 0))


def _texture_height(texture: Any) -> int:
    if isinstance(texture, Mapping):
        return int(texture.get("height", 0))
    return int(getattr(texture, "height", 0))


def _texture_channel(texture: Any) -> int:
    if isinstance(texture, Mapping):
        return int(texture.get("channel", 0))
    return int(getattr(texture, "channel", 0))


def _texture_encoding(texture: Any) -> str:
    if isinstance(texture, Mapping):
        value = texture.get("encoding", "")
    else:
        value = getattr(texture, "encoding", "")
    return "" if value is None else str(value).strip().lower()


def _texture_scale_values(texture: Any, channels: int, default: Sequence[float]) -> np.ndarray:
    if isinstance(texture, Mapping):
        scale = texture.get("scale", None)
    else:
        scale = getattr(texture, "scale", None)
    if scale is None:
        values = np.asarray(default, dtype=np.float32)
    else:
        values = np.asarray(scale, dtype=np.float32).reshape(-1)
    if values.size == 1 and channels > 1:
        values = np.repeat(values, channels)
    elif values.size < channels:
        pad = np.asarray(default, dtype=np.float32)
        values = np.concatenate([values, pad[values.size : channels]])
    else:
        values = values[:channels]
    return np.clip(values.astype(np.float32), 0.0, 1.0)


def _resolve_texture_file_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    repo_candidate = _repo_root() / candidate
    if repo_candidate.exists():
        return repo_candidate
    return candidate


def _infer_texture_mime_type(texture: Any, role: str) -> str:
    encoding = _texture_encoding(texture)
    suffix = Path(_texture_file(texture)).suffix.lower()
    if encoding in ("png", "image/png") or suffix == ".png":
        return "image/png"
    if encoding in ("jpg", "jpeg", "image/jpeg") or suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    raise ValueError(f"{role} only supports PNG/JPEG encoded textures when using file/image bytes.")


def _normalize_raw_texture_pixels(texture: Any, role: str) -> np.ndarray:
    image_data = _texture_image_data(texture)
    width = _texture_width(texture)
    height = _texture_height(texture)
    channels = _texture_channel(texture)
    if width <= 0 or height <= 0:
        raise ValueError(f"{role} raw texture requires positive width and height.")
    if channels <= 0:
        pixel_count = width * height
        if pixel_count <= 0 or len(image_data) % pixel_count != 0:
            raise ValueError(f"{role} raw texture must provide channel count or exact image_data size.")
        channels = len(image_data) // pixel_count
    if channels not in (1, 2, 3, 4):
        raise ValueError(f"{role} raw texture only supports 1-4 channels, got {channels}.")
    expected_size = width * height * channels
    if len(image_data) != expected_size:
        raise ValueError(f"{role} raw texture expects {expected_size} bytes, got {len(image_data)}.")
    return np.frombuffer(image_data, dtype=np.uint8).reshape(height, width, channels).copy()


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
    color_type = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    raw_scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw_scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _build_texture_desc(texture: Any, role: str, default_name: str) -> EmbeddedTextureDesc:
    image_data = _texture_image_data(texture)
    if image_data:
        encoding = _texture_encoding(texture)
        if encoding in ("", "raw", "raw8", "r8", "rg8", "rgb8", "rgba8"):
            pixels = _normalize_raw_texture_pixels(texture, role)
            return EmbeddedTextureDesc(
                name=default_name,
                image_bytes=_encode_png(pixels),
                mime_type="image/png",
                has_alpha=int(pixels.shape[2]) in (2, 4),
            )
        return EmbeddedTextureDesc(
            name=default_name,
            image_bytes=image_data,
            mime_type=_infer_texture_mime_type(texture, role),
            has_alpha=_texture_channel(texture) in (2, 4),
        )

    file_value = _texture_file(texture)
    if not file_value:
        raise ValueError(f"{role} texture payload is empty.")
    path = _resolve_texture_file_path(file_value)
    if not path.is_file():
        raise ValueError(f"{role} texture file was not found: {path}")
    return EmbeddedTextureDesc(
        name=default_name,
        image_bytes=path.read_bytes(),
        mime_type=_infer_texture_mime_type(texture, role),
        has_alpha=_texture_channel(texture) in (2, 4),
    )


def _raw_texture_size(texture: Any, role: str) -> tuple[int, int]:
    pixels = _normalize_raw_texture_pixels(texture, role)
    return int(pixels.shape[0]), int(pixels.shape[1])


def _raw_texture_to_channels(texture: Any, role: str, size: tuple[int, int], channels: int, default: Sequence[int]) -> np.ndarray:
    height, width = size
    if texture is None:
        fill = np.asarray(default[:channels], dtype=np.uint8).reshape(1, 1, channels)
        return np.broadcast_to(fill, (height, width, channels)).copy()
    pixels = _normalize_raw_texture_pixels(texture, role)
    if (pixels.shape[0], pixels.shape[1]) != size:
        raise ValueError(f"{role} raw textures must share the same resolution.")
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


def _build_base_color_texture(base_texture: Any, opacity_texture: Any, name: str) -> Optional[EmbeddedTextureDesc]:
    if not _texture_has_payload(base_texture) and not _texture_has_payload(opacity_texture):
        return None
    if _texture_has_payload(opacity_texture) and _texture_encoding(opacity_texture) not in ("", "raw", "raw8", "r8", "rg8", "rgb8", "rgba8"):
        raise ValueError("Opacity textures currently require raw image_data so they can be packed into base color alpha.")
    if _texture_has_payload(base_texture) and _texture_encoding(base_texture) not in ("", "raw", "raw8", "r8", "rg8", "rgb8", "rgba8") and _texture_has_payload(opacity_texture):
        raise ValueError("Encoded/file base color textures cannot currently be combined with a separate opacity texture.")
    if _texture_has_payload(base_texture) and _texture_encoding(base_texture) not in ("", "raw", "raw8", "r8", "rg8", "rgb8", "rgba8"):
        return _build_texture_desc(base_texture, f"{name}.base_color", f"{name}_base_color")

    size = None
    if _texture_has_payload(base_texture):
        size = _raw_texture_size(base_texture, f"{name}.base_color")
    if _texture_has_payload(opacity_texture):
        opacity_size = _raw_texture_size(opacity_texture, f"{name}.opacity")
        if size is None:
            size = opacity_size
        elif size != opacity_size:
            raise ValueError("Base color and opacity textures must share the same resolution.")
    if size is None:
        return None

    rgba = _raw_texture_to_channels(base_texture, f"{name}.base_color", size, 4, (255, 255, 255, 255))
    if _texture_has_payload(opacity_texture):
        alpha = _raw_texture_to_channels(opacity_texture, f"{name}.opacity", size, 1, (255,))
        rgba[:, :, 3] = alpha[:, :, 0]
    return EmbeddedTextureDesc(
        name=f"{name}_base_color",
        image_bytes=_encode_png(rgba),
        mime_type="image/png",
        has_alpha=True,
    )


def _build_metallic_roughness_texture(roughness_texture: Any, metallic_texture: Any, name: str) -> Optional[EmbeddedTextureDesc]:
    if not _texture_has_payload(roughness_texture) and not _texture_has_payload(metallic_texture):
        return None
    for role, texture in (("roughness", roughness_texture), ("metallic", metallic_texture)):
        if _texture_has_payload(texture) and _texture_encoding(texture) not in ("", "raw", "raw8", "r8", "rg8", "rgb8", "rgba8"):
            raise ValueError(f"{name}.{role} currently requires raw image_data so it can be packed into metallicRoughnessTexture.")

    size = None
    if _texture_has_payload(roughness_texture):
        size = _raw_texture_size(roughness_texture, f"{name}.roughness")
    if _texture_has_payload(metallic_texture):
        metallic_size = _raw_texture_size(metallic_texture, f"{name}.metallic")
        if size is None:
            size = metallic_size
        elif size != metallic_size:
            raise ValueError("Roughness and metallic textures must share the same resolution.")
    if size is None:
        return None

    packed = np.zeros((size[0], size[1], 4), dtype=np.uint8)
    packed[:, :, 1] = _raw_texture_to_channels(roughness_texture, f"{name}.roughness", size, 1, (255,))[:, :, 0]
    packed[:, :, 2] = _raw_texture_to_channels(metallic_texture, f"{name}.metallic", size, 1, (255,))[:, :, 0]
    packed[:, :, 3] = 255
    return EmbeddedTextureDesc(
        name=f"{name}_metallic_roughness",
        image_bytes=_encode_png(packed),
        mime_type="image/png",
        has_alpha=False,
    )


def _build_emissive_texture(texture: Any, name: str) -> Optional[EmbeddedTextureDesc]:
    if not _texture_has_payload(texture):
        return None
    if _texture_encoding(texture) not in ("", "raw", "raw8", "r8", "rg8", "rgb8", "rgba8"):
        return _build_texture_desc(texture, f"{name}.emissive", f"{name}_emissive")
    size = _raw_texture_size(texture, f"{name}.emissive")
    rgb = _raw_texture_to_channels(texture, f"{name}.emissive", size, 3, (255, 255, 255))
    return EmbeddedTextureDesc(
        name=f"{name}_emissive",
        image_bytes=_encode_png(rgb),
        mime_type="image/png",
        has_alpha=False,
    )


def _coerce_surface_desc(surface: Any) -> "SurfaceDesc":
    if isinstance(surface, SurfaceDesc):
        return surface

    if surface is None:
        return SurfaceDesc()

    if isinstance(surface, Mapping):
        base = surface.get("base_color", surface.get("color", (0.8, 0.8, 0.8, 1.0)))
        if "opacity" in surface and len(base) < 4:
            base = tuple(base) + (surface["opacity"],)
        emissive = surface.get("emissive", (0.0, 0.0, 0.0))
        material_name = str(surface.get("name", "surface"))
        base_color_texture = _build_base_color_texture(surface.get("base_color"), surface.get("opacity"), material_name)
        metallic_roughness_texture = _build_metallic_roughness_texture(
            surface.get("roughness"),
            surface.get("metallic"),
            material_name,
        )
        emissive_texture = _build_emissive_texture(surface.get("emissive"), material_name)
        return SurfaceDesc(
            base_color=tuple(_extract_texture_value(base, 4, (0.8, 0.8, 0.8, 1.0), textured_default=(1.0, 1.0, 1.0, 1.0))),
            roughness=float(_extract_texture_value(surface.get("roughness", 1.0), 1, (1.0,), textured_default=(1.0,))[0]),
            metallic=float(_extract_texture_value(surface.get("metallic", 0.0), 1, (0.0,), textured_default=(1.0,))[0]),
            emissive=tuple(_extract_texture_value(emissive, 3, (0.0, 0.0, 0.0), textured_default=(1.0, 1.0, 1.0))),
            double_sided=bool(surface.get("double_sided", False)),
            base_color_texture=base_color_texture,
            metallic_roughness_texture=metallic_roughness_texture,
            emissive_texture=emissive_texture,
        )

    rgba_source = surface.get_rgba() if hasattr(surface, "get_rgba") else getattr(surface, "color", None)
    emission_source = surface.get_emission() if hasattr(surface, "get_emission") else getattr(surface, "emissive", None)
    roughness_source = getattr(surface, "roughness_texture", getattr(surface, "roughness", None))
    metallic_source = getattr(surface, "metallic_texture", getattr(surface, "metallic", None))
    material_name = str(getattr(surface, "name", "surface"))

    return SurfaceDesc(
        base_color=tuple(
            _extract_texture_value(
                rgba_source,
                4,
                (0.8, 0.8, 0.8, 1.0),
                textured_default=(1.0, 1.0, 1.0, 1.0),
            )
        ),
        roughness=float(_extract_texture_value(roughness_source, 1, (1.0,), textured_default=(1.0,))[0]),
        metallic=float(_extract_texture_value(metallic_source, 1, (0.0,), textured_default=(1.0,))[0]),
        emissive=tuple(_extract_texture_value(emission_source, 3, (0.0, 0.0, 0.0), textured_default=(1.0, 1.0, 1.0))),
        double_sided=bool(getattr(surface, "double_sided", False) or False),
        base_color_texture=_build_base_color_texture(
            rgba_source,
            getattr(surface, "opacity", None),
            material_name,
        ),
        metallic_roughness_texture=_build_metallic_roughness_texture(
            roughness_source,
            metallic_source,
            material_name,
        ),
        emissive_texture=_build_emissive_texture(emission_source, material_name),
    )


def _coerce_camera_desc(camera: Any) -> "CameraDesc":
    if isinstance(camera, CameraDesc):
        return camera

    if isinstance(camera, Mapping):
        source = camera
        uid = str(source.get("uid", source.get("name", "camera")))
        model = str(source.get("model", "pinhole"))
        pos = source.get("pos", source.get("position"))
        lookat = source.get("lookat", source.get("target"))
        up = source.get("up", (0.0, 1.0, 0.0))
        res = source.get("res", source.get("resolution", (512, 512)))
        fov = float(source.get("fov", 45.0))
        near = float(source.get("near", 0.1))
        far = float(source.get("far", 1000.0))
    else:
        uid = str(getattr(camera, "uid", getattr(camera, "name", "camera")))
        model = str(getattr(camera, "model", "pinhole"))
        pos = getattr(camera, "pos", getattr(camera, "position", None))
        lookat = getattr(camera, "lookat", getattr(camera, "target", None))
        up = getattr(camera, "up", (0.0, 1.0, 0.0))
        res = getattr(camera, "res", getattr(camera, "resolution", (512, 512)))
        fov = float(getattr(camera, "fov", 45.0))
        near = float(getattr(camera, "near", 0.1))
        far = float(getattr(camera, "far", 1000.0))

    if pos is None or lookat is None:
        raise ValueError("Camera must provide both position/pos and target/lookat.")
    if model != "pinhole":
        raise ValueError("GenesisStyleRenderer MVP-1 only supports pinhole cameras.")

    res_array = np.asarray(res, dtype=np.int32).reshape(-1)
    if res_array.size != 2:
        raise ValueError("Camera resolution must contain two integers.")

    return CameraDesc(
        uid=uid,
        model=model,
        pos=tuple(_as_float_array(pos, 3)),
        lookat=tuple(_as_float_array(lookat, 3)),
        up=tuple(_as_float_array(up, 3)),
        res=(int(res_array[0]), int(res_array[1])),
        fov=fov,
        near=near,
        far=far,
    )


def _make_unit_sphere(latitudes: int, longitudes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    uvs: list[list[float]] = []
    triangles: list[list[int]] = []

    for lat in range(latitudes + 1):
        v = lat / latitudes
        phi = np.pi * v
        sin_phi = float(np.sin(phi))
        cos_phi = float(np.cos(phi))

        for lon in range(longitudes + 1):
            u = lon / longitudes
            theta = 2.0 * np.pi * u
            sin_theta = float(np.sin(theta))
            cos_theta = float(np.cos(theta))

            x = sin_phi * cos_theta
            y = cos_phi
            z = sin_phi * sin_theta
            normals.append([x, y, z])
            vertices.append([x, y, z])
            uvs.append([u, 1.0 - v])

    row = longitudes + 1
    for lat in range(latitudes):
        for lon in range(longitudes):
            a = lat * row + lon
            b = a + row
            c = a + 1
            d = b + 1
            if lat != 0:
                triangles.append([a, b, c])
            if lat != latitudes - 1:
                triangles.append([c, b, d])

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(triangles, dtype=np.uint32),
        np.asarray(uvs, dtype=np.float32),
    )


class _RuntimeHandle:
    _lock = threading.Lock()
    _module = None
    _options = None
    _refcount = 0

    @classmethod
    def acquire(
        cls,
        module_dir: Path,
        runtime_dir: Path,
        backend: str,
        device_index: int,
        enable_debug: bool,
        enable_external_interop: bool,
    ):
        module_dir = module_dir.resolve()
        runtime_dir = runtime_dir.resolve()
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))

        module = _import_native_renderer_module()
        options = (
            module.__name__,
            str(module_dir),
            str(runtime_dir),
            backend,
            int(device_index),
            bool(enable_debug),
            bool(enable_external_interop),
        )

        with cls._lock:
            if cls._refcount == 0:
                module.init(
                    runtime_dir=str(runtime_dir),
                    backend=backend,
                    device_index=int(device_index),
                    enable_debug=bool(enable_debug),
                    enable_external_interop=bool(enable_external_interop),
                )
                cls._module = module
                cls._options = options
            elif cls._options != options:
                raise RuntimeError(f"{cls._module.__name__} is already initialized with different options.")

            cls._refcount += 1
            return cls._module

    @classmethod
    def release(cls) -> None:
        with cls._lock:
            if cls._refcount == 0:
                return

            cls._refcount -= 1
            if cls._refcount == 0 and cls._module is not None:
                cls._module.destroy()
                cls._module = None
                cls._options = None


@dataclass(frozen=True)
class SurfaceDesc:
    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    roughness: float = 1.0
    metallic: float = 0.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    double_sided: bool = False
    base_color_texture: Optional[EmbeddedTextureDesc] = None
    metallic_roughness_texture: Optional[EmbeddedTextureDesc] = None
    emissive_texture: Optional[EmbeddedTextureDesc] = None


@dataclass(frozen=True)
class CameraDesc:
    uid: str
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    res: tuple[int, int] = (512, 512)
    fov: float = 45.0
    near: float = 0.1
    far: float = 1000.0
    model: str = "pinhole"


@dataclass
class _ShapeRecord:
    kind: str
    surface_name: str
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.uint32))
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    uvs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    particle_radius: float = 0.05
    particle_centers: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    particle_radii: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))


class GenesisStyleRenderer:
    def __init__(
        self,
        module_dir: Optional[Path | str] = None,
        runtime_dir: Optional[Path | str] = None,
        backend: str = "vulkan",
        device_index: int = -1,
        enable_debug: bool = False,
        rendered_envs_idx: Optional[Sequence[int]] = None,
        particle_sphere_segments: tuple[int, int] = (10, 20),
        enable_external_interop: bool = False,
    ) -> None:
        self.module_dir = Path(module_dir) if module_dir is not None else _default_module_dir()
        self.runtime_dir = Path(runtime_dir) if runtime_dir is not None else _default_runtime_dir()
        self.rendered_envs_idx = list(rendered_envs_idx) if rendered_envs_idx is not None else [0]

        self._rr = _RuntimeHandle.acquire(
            module_dir=self.module_dir,
            runtime_dir=self.runtime_dir,
            backend=backend,
            device_index=device_index,
            enable_debug=enable_debug,
            enable_external_interop=enable_external_interop,
        )
        self._scene = self._rr.create_scene()

        self._surfaces: dict[str, SurfaceDesc] = {}
        self._shapes: dict[str, _ShapeRecord] = {}
        self._cameras: dict[str, CameraDesc] = {}
        self._scene_dirty = True
        self._scene_loaded = False
        self._camera_dirty = False
        self.camera_updated = False
        self._last_camera_uid: Optional[str] = None
        self._pending_rigid_updates: dict[str, np.ndarray] = {}
        self._rigid_node_handles: dict[str, int] = {}
        self._native_camera_indices: dict[str, int] = {}
        self._t = -1
        self._destroyed = False

        self._ambient_top = (0.03, 0.04, 0.06)
        self._ambient_bottom = (0.01, 0.01, 0.01)
        self._default_light_direction = (-0.4, -1.0, -0.6)
        self._default_light_color = (1.0, 1.0, 1.0)
        self._default_light_irradiance = 2.0
        self._scene.set_ambient(self._ambient_top, self._ambient_bottom)
        self._scene.set_default_light(
            self._default_light_direction,
            self._default_light_color,
            self._default_light_irradiance,
        )

        latitudes, longitudes = particle_sphere_segments
        self._sphere_vertices, self._sphere_triangles, self._sphere_uvs = _make_unit_sphere(
            max(3, int(latitudes)),
            max(3, int(longitudes)),
        )

        self._temp_dir = _repo_root() / ".temp" / "rtxns_genesis_style"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._scene_path = self._temp_dir / "scene.glb"

    def create_shared_transform_stream(self, record_count: int, slot_count: int = 3):
        """Create a Vulkan/CUDA shared transform stream on this renderer context."""
        return self._rr.create_shared_transform_stream(
            record_count=int(record_count),
            slot_count=int(slot_count),
        )

    def __enter__(self) -> "GenesisStyleRenderer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

    def add_surface(self, shape_name: str, surface: Any) -> SurfaceDesc:
        desc = _coerce_surface_desc(surface)
        self._surfaces[str(shape_name)] = desc
        self._scene_dirty = True
        return desc

    def update_surface(self, shape_name: str, surface: Any) -> SurfaceDesc:
        return self.add_surface(shape_name, surface)

    def add_rigid(
        self,
        name: str,
        vertices: Any,
        triangles: Any,
        normals: Any = None,
        uvs: Any = None,
        batch_index: Optional[int] = None,
    ) -> None:
        shape_name = _shape_name(str(name), batch_index)
        vertex_array = _coerce_vertices(vertices)
        triangle_array = _coerce_triangles(triangles)
        normal_array = _coerce_normals(normals, vertex_array.shape[0])
        if normal_array.size == 0:
            normal_array = _compute_vertex_normals(vertex_array, triangle_array)
        uv_array = _coerce_uvs(uvs, vertex_array.shape[0])

        self._shapes[shape_name] = _ShapeRecord(
            kind="rigid",
            surface_name=str(name),
            vertices=vertex_array,
            triangles=triangle_array,
            normals=normal_array,
            uvs=uv_array,
        )
        self._scene_dirty = True

    def add_rigid_batch(self, name: str, vertices: Any, triangles: Any, normals: Any = None, uvs: Any = None) -> None:
        for batch_index in self.rendered_envs_idx:
            self.add_rigid(name, vertices, triangles, normals, uvs, batch_index=batch_index)

    def update_rigid(self, name: str, matrix: Any, batch_index: Optional[int] = None) -> None:
        shape_name = _shape_name(str(name), batch_index)
        if shape_name not in self._shapes:
            raise KeyError(f"Rigid shape '{shape_name}' has not been added.")
        record = self._shapes[shape_name]
        if record.kind != "rigid":
            raise ValueError(f"Shape '{shape_name}' is not rigid.")
        record.transform = _coerce_transform(matrix)
        if self._scene_loaded and not self._scene_dirty:
            self._pending_rigid_updates[shape_name] = np.ascontiguousarray(record.transform)
        else:
            self._scene_dirty = True

    def update_rigid_batch(self, name: str, matrices: Any) -> None:
        for batch_index in self.rendered_envs_idx:
            self.update_rigid(name, matrices[batch_index], batch_index=batch_index)

    def add_deformable(self, name: str, batch_index: Optional[int] = None) -> None:
        shape_name = _shape_name(str(name), batch_index)
        self._shapes[shape_name] = _ShapeRecord(
            kind="deformable",
            surface_name=str(name),
        )
        self._scene_dirty = True

    def update_deformable(
        self,
        name: str,
        vertices: Any,
        triangles: Any,
        normals: Any = None,
        uvs: Any = None,
        batch_index: Optional[int] = None,
    ) -> None:
        shape_name = _shape_name(str(name), batch_index)
        if shape_name not in self._shapes:
            self.add_deformable(name, batch_index=batch_index)
        record = self._shapes[shape_name]
        vertex_array = _coerce_vertices(vertices)
        triangle_array = _coerce_triangles(triangles)
        normal_array = _coerce_normals(normals, vertex_array.shape[0])
        if normal_array.size == 0:
            normal_array = _compute_vertex_normals(vertex_array, triangle_array)
        uv_array = _coerce_uvs(uvs, vertex_array.shape[0])

        _validate_finite_xyz(vertex_array, label="Deformable vertices")
        _validate_triangle_indices(triangle_array, vertex_array.shape[0])

        record.kind = "deformable"
        record.vertices = vertex_array
        record.triangles = triangle_array
        record.normals = normal_array
        record.uvs = uv_array
        record.transform = np.eye(4, dtype=np.float32)
        self._scene_dirty = True

    def add_particles(self, name: str, radius: Optional[float] = None, density: Optional[float] = None) -> None:
        del density
        self._shapes[str(name)] = _ShapeRecord(
            kind="particles",
            surface_name=str(name),
            particle_radius=float(radius if radius is not None else 0.05),
        )
        self._scene_dirty = True

    def update_particles(
        self,
        name: str,
        particles: Any,
        radius: Optional[float] = None,
        particles_vel: Any = None,
        particles_radii: Any = None,
    ) -> None:
        del particles_vel
        shape_name = str(name)
        if shape_name not in self._shapes:
            self.add_particles(name, radius=radius)

        centers = _coerce_vertices(particles)
        _validate_finite_xyz(centers, label="Particle centers")
        record = self._shapes[shape_name]
        if particles_radii is None:
            particle_radius = float(radius if radius is not None else record.particle_radius)
            radii = np.full((centers.shape[0],), particle_radius, dtype=np.float32)
        else:
            radii = np.asarray(particles_radii, dtype=np.float32).reshape(-1)
            if radii.size != centers.shape[0]:
                raise ValueError("particles_radii must have the same length as particles.")
            particle_radius = float(radius if radius is not None else (float(radii[0]) if radii.size > 0 else 0.05))

        radii = np.clip(np.asarray(radii, dtype=np.float32), _PARTICLE_RADIUS_MIN, _PARTICLE_RADIUS_MAX)
        if not np.isfinite(radii).all():
            raise ValueError("Particle radii must be finite.")

        record.kind = "particles"
        record.particle_centers = np.ascontiguousarray(centers)
        record.particle_radii = np.ascontiguousarray(radii.astype(np.float32))
        record.particle_radius = particle_radius
        self._scene_dirty = True

    def add_camera(self, camera: Any) -> CameraDesc:
        desc = _coerce_camera_desc(camera)
        self._cameras[desc.uid] = desc
        self._last_camera_uid = desc.uid
        self._camera_dirty = True
        self.camera_updated = True
        return desc

    def update_camera(self, camera: Any) -> CameraDesc:
        return self.add_camera(camera)

    def set_ambient(self, top_rgb: Sequence[float], bottom_rgb: Sequence[float]) -> None:
        self._ambient_top = tuple(_as_float_array(top_rgb, 3))
        self._ambient_bottom = tuple(_as_float_array(bottom_rgb, 3))
        self._scene.set_ambient(self._ambient_top, self._ambient_bottom)

    def set_default_light(
        self,
        direction: Sequence[float],
        color: Sequence[float] = (1.0, 1.0, 1.0),
        irradiance: float = 2.0,
    ) -> None:
        self._default_light_direction = tuple(_as_float_array(direction, 3))
        self._default_light_color = tuple(_as_float_array(color, 3))
        self._default_light_irradiance = float(irradiance)
        self._scene.set_default_light(
            self._default_light_direction,
            self._default_light_color,
            self._default_light_irradiance,
        )

    def reset(self) -> None:
        self._t = -1

    def update_scene(self, force_render: bool = False, time: Optional[float] = None) -> None:
        if not force_render and not self._scene_dirty:
            self._apply_pending_incremental_updates()
            if time is not None:
                self._t = time
            self.camera_updated = False
            return

        builder = GlbSceneBuilder()
        material_indices: dict[str, int] = {}
        renderable_count = 0
        rigid_node_names: list[str] = []
        sensor_node_names: list[str] = []

        for shape_name, record in self._shapes.items():
            vertices, triangles, normals, uvs = self._shape_to_mesh(record)
            if vertices.size == 0 or triangles.size == 0:
                continue

            material_name = record.surface_name
            if material_name not in material_indices:
                surface = self._surfaces.get(material_name, SurfaceDesc())
                material_indices[material_name] = builder.add_material(
                    name=material_name,
                    base_color=np.asarray(surface.base_color, dtype=np.float32),
                    roughness=float(surface.roughness),
                    metallic=float(surface.metallic),
                    emissive=np.asarray(surface.emissive, dtype=np.float32),
                    double_sided=bool(surface.double_sided),
                    base_color_texture=surface.base_color_texture,
                    metallic_roughness_texture=surface.metallic_roughness_texture,
                    emissive_texture=surface.emissive_texture,
                )

            builder.add_mesh(
                name=shape_name,
                vertices=vertices,
                triangles=triangles,
                normals=normals,
                uvs=uvs,
                material_index=material_indices[material_name],
                node_matrix=_transform_to_gltf_node_matrix(record.transform) if record.kind == "rigid" else None,
            )
            if record.kind == "rigid":
                rigid_node_names.append(shape_name)
            sensor_node_names.append(shape_name)
            renderable_count += 1

        if renderable_count == 0:
            raise RuntimeError("update_scene() found no renderable shapes. Add geometry before rendering.")

        self._scene_path.write_bytes(builder.build())
        self._scene.load_scene(str(self._scene_path))
        self._scene_loaded = True
        self._pending_rigid_updates.clear()
        self._rigid_node_handles.clear()
        if rigid_node_names and hasattr(self._scene, "get_node_handles"):
            handles = self._scene.get_node_handles(rigid_node_names)
            self._rigid_node_handles.update(zip(rigid_node_names, handles))
        if sensor_node_names and hasattr(self._scene, "set_node_labels"):
            self._scene.set_node_labels(
                sensor_node_names,
                list(range(1, len(sensor_node_names) + 1)),
                [0] * len(sensor_node_names),
            )
        self._scene.set_ambient(self._ambient_top, self._ambient_bottom)
        self._scene.set_default_light(
            self._default_light_direction,
            self._default_light_color,
            self._default_light_irradiance,
        )
        self._apply_latest_camera()

        self._scene_dirty = False
        self._camera_dirty = False
        self.camera_updated = False
        self._t = self._t + 1 if time is None else time

    def render_camera(self, camera: Any, force_render: bool = False, time: Optional[float] = None) -> np.ndarray:
        desc = _coerce_camera_desc(camera)
        self._cameras[desc.uid] = desc
        self._last_camera_uid = desc.uid
        self.update_scene(force_render=force_render, time=time)
        self._apply_camera_desc(desc)
        self._camera_dirty = False
        rgba = self._scene.render_frame()
        image = np.frombuffer(rgba, dtype=np.uint8).reshape(desc.res[1], desc.res[0], 4)
        return np.ascontiguousarray(image[:, :, :3])

    def render_frame(self, camera: Any, force_render: bool = False, time: Optional[float] = None) -> np.ndarray:
        return self.render_camera(camera, force_render=force_render, time=time)

    def render_camera_rgba(self, camera: Any, force_render: bool = False, time: Optional[float] = None) -> np.ndarray:
        desc = _coerce_camera_desc(camera)
        self._cameras[desc.uid] = desc
        self._last_camera_uid = desc.uid
        self.update_scene(force_render=force_render, time=time)
        self._apply_camera_desc(desc)
        self._camera_dirty = False
        rgba = self._scene.render_frame()
        return np.frombuffer(rgba, dtype=np.uint8).reshape(desc.res[1], desc.res[0], 4).copy()

    def render_sensor(
        self,
        camera: Any,
        products: Sequence[str] = SENSOR_PRODUCTS,
        force_render: bool = False,
        time: Optional[float] = None,
    ) -> SensorFrame:
        return self.render_sensor_batch(
            [camera],
            products=products,
            force_render=force_render,
            time=time,
        )[0]

    def render_sensor_batch(
        self,
        cameras: Sequence[Any],
        products: Sequence[str] = SENSOR_PRODUCTS,
        force_render: bool = False,
        time: Optional[float] = None,
    ) -> tuple[SensorFrame, ...]:
        descs = tuple(_coerce_camera_desc(camera) for camera in cameras)
        if not descs:
            return ()
        if len({desc.uid for desc in descs}) != len(descs):
            raise ValueError("A sensor batch cannot contain duplicate camera UIDs.")
        normalized_products = normalize_sensor_products(products)
        for desc in descs:
            self._cameras[desc.uid] = desc
        self._last_camera_uid = descs[-1].uid
        self.update_scene(force_render=force_render, time=time)
        camera_indices = [self._ensure_native_camera_desc(desc) for desc in descs]
        raw_frames = self._scene.render_sensor_batch(
            camera_indices,
            list(normalized_products),
        )
        self._camera_dirty = False
        self.camera_updated = False
        return decode_sensor_frames(raw_frames)

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True

        self._cameras.clear()
        self._camera_dirty = False
        self._last_camera_uid = None
        self._pending_rigid_updates.clear()
        self._rigid_node_handles.clear()
        self._native_camera_indices.clear()
        self._shapes.clear()
        self._surfaces.clear()
        self._scene_loaded = False
        self._scene = None

        shutil.rmtree(self._temp_dir, ignore_errors=True)
        _RuntimeHandle.release()

    @property
    def cameras(self) -> dict[str, CameraDesc]:
        return dict(self._cameras)

    def _shape_to_mesh(self, record: _ShapeRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if record.kind == "particles":
            return self._particles_to_mesh(record)

        vertices = record.vertices
        triangles = record.triangles
        normals = record.normals
        uvs = record.uvs
        if vertices.size == 0 or triangles.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )

        return (
            np.ascontiguousarray(vertices, dtype=np.float32),
            np.ascontiguousarray(triangles, dtype=np.uint32),
            np.ascontiguousarray(normals, dtype=np.float32),
            np.ascontiguousarray(uvs, dtype=np.float32),
        )

    def _apply_camera_desc(self, desc: CameraDesc) -> None:
        for uid, index in tuple(self._native_camera_indices.items()):
            if index == 0 and uid != desc.uid:
                del self._native_camera_indices[uid]
        self._scene.set_camera(
            desc.pos,
            desc.lookat,
            desc.up,
            float(desc.fov),
            int(desc.res[0]),
            int(desc.res[1]),
            float(desc.near),
            float(desc.far),
        )
        self._native_camera_indices[desc.uid] = 0

    def _ensure_native_camera_desc(self, desc: CameraDesc) -> int:
        camera_index = self._native_camera_indices.get(desc.uid)
        camera_args = (
            desc.pos,
            desc.lookat,
            desc.up,
            float(desc.fov),
            int(desc.res[0]),
            int(desc.res[1]),
            float(desc.near),
            float(desc.far),
        )
        if camera_index is None:
            claimed_indices = set(self._native_camera_indices.values())
            reusable_index = next(
                (
                    index
                    for index in range(int(self._scene.camera_count))
                    if index not in claimed_indices
                ),
                None,
            )
            if reusable_index is not None:
                camera_index = reusable_index
                self._scene.set_camera_at(camera_index, *camera_args)
            else:
                camera_index = int(self._scene.add_camera(*camera_args))
            self._native_camera_indices[desc.uid] = camera_index
        else:
            self._scene.set_camera_at(camera_index, *camera_args)
        return camera_index

    def _apply_latest_camera(self) -> None:
        if self._last_camera_uid is None:
            return
        desc = self._cameras.get(self._last_camera_uid)
        if desc is None:
            return
        self._apply_camera_desc(desc)

    def _apply_pending_incremental_updates(self) -> None:
        if self._pending_rigid_updates:
            shape_names = list(self._pending_rigid_updates)
            if hasattr(self._scene, "update_node_transforms_batch"):
                missing = [
                    name for name in shape_names if name not in self._rigid_node_handles
                ]
                if missing:
                    handles = self._scene.get_node_handles(missing)
                    self._rigid_node_handles.update(zip(missing, handles))
                self._scene.update_node_transforms_batch(
                    [self._rigid_node_handles[name] for name in shape_names],
                    [
                        np.asarray(
                            self._pending_rigid_updates[name], dtype=np.float32
                        ).reshape(-1).tolist()
                        for name in shape_names
                    ],
                )
            else:
                for shape_name in shape_names:
                    matrix = self._pending_rigid_updates[shape_name]
                    self._scene.update_node_transform(
                        shape_name,
                        np.asarray(matrix, dtype=np.float32).reshape(-1).tolist(),
                    )
            self._pending_rigid_updates.clear()
        if self._camera_dirty:
            self._apply_latest_camera()
            self._camera_dirty = False

    def _particles_to_mesh(self, record: _ShapeRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        centers = record.particle_centers
        radii = record.particle_radii
        if centers.size == 0 or radii.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )

        template_vertices = self._sphere_vertices
        template_triangles = self._sphere_triangles
        template_normals = template_vertices
        template_uvs = self._sphere_uvs

        vertex_count = template_vertices.shape[0]
        face_count = template_triangles.shape[0]
        all_vertices = np.empty((centers.shape[0] * vertex_count, 3), dtype=np.float32)
        all_normals = np.empty_like(all_vertices)
        all_uvs = np.tile(template_uvs, (centers.shape[0], 1)).astype(np.float32)
        all_triangles = np.empty((centers.shape[0] * face_count, 3), dtype=np.uint32)

        for index, (center, radius) in enumerate(zip(centers, radii, strict=False)):
            vertex_offset = index * vertex_count
            face_offset = index * face_count
            all_vertices[vertex_offset : vertex_offset + vertex_count] = template_vertices * radius + center
            all_normals[vertex_offset : vertex_offset + vertex_count] = template_normals
            all_triangles[face_offset : face_offset + face_count] = template_triangles + vertex_offset

        return all_vertices, all_triangles, all_normals, all_uvs
