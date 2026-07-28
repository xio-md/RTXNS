from __future__ import annotations

import argparse
import importlib
import math
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from rtxns_genesis_style.glb_builder import GlbSceneBuilder


HEADER_MAGIC = 0x31545346
HEADER_SIZE = 64
RECORD_STRIDE = 48
NODE_NAMES = ("dynamic_a", "dynamic_b")


def _build_scene(path: Path) -> None:
    builder = GlbSceneBuilder()
    material = builder.add_material(
        name="synthetic",
        base_color=np.asarray((0.8, 0.4, 0.2, 1.0), dtype=np.float32),
        roughness=0.8,
        metallic=0.0,
        emissive=np.zeros(3, dtype=np.float32),
        double_sided=False,
    )
    vertices = np.asarray(
        ((-0.1, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.2, 0.0)),
        dtype=np.float32,
    )
    triangles = np.asarray(((0, 1, 2),), dtype=np.uint32)
    normals = np.asarray(((0.0, 0.0, 1.0),) * 3, dtype=np.float32)
    uvs = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)), dtype=np.float32)
    for name in NODE_NAMES:
        builder.add_mesh(
            name=name,
            vertices=vertices,
            triangles=triangles,
            normals=normals,
            uvs=uvs,
            material_index=material,
        )
    path.write_bytes(builder.build())


def _matrices(epoch: int, slot: int) -> list[list[float]]:
    angle_a = 0.17 * epoch + 0.03 * slot
    ca, sa = math.cos(angle_a), math.sin(angle_a)
    matrix_a = [
        ca,
        -sa,
        0.13,
        0.4 * epoch + 0.07 * slot,
        sa,
        ca,
        -0.21,
        -0.3 * epoch + 0.05 * slot,
        0.09,
        0.18,
        1.0,
        0.2 * epoch - 0.04 * slot,
        0.0,
        0.0,
        0.0,
        1.0,
    ]

    angle_b = -0.11 * epoch + 0.02 * slot
    cb, sb = math.cos(angle_b), math.sin(angle_b)
    matrix_b = [
        cb,
        0.16,
        sb,
        -0.25 * epoch + 0.03 * slot,
        -0.12,
        1.0,
        0.08,
        0.15 * epoch - 0.02 * slot,
        -sb,
        0.19,
        cb,
        0.35 * epoch + 0.06 * slot,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    return [matrix_a, matrix_b]


def _payload(
    matrices: list[list[float]],
    *,
    epoch: int,
    slot: int,
    magic: int = HEADER_MAGIC,
    abi_major: int = 1,
    abi_minor: int = 0,
    header_size: int = HEADER_SIZE,
    header_record_count: int | None = None,
) -> bytes:
    record_count = len(matrices)
    payload = bytearray(HEADER_SIZE + record_count * RECORD_STRIDE)
    struct.pack_into(
        "<IHHIIQQQI",
        payload,
        0,
        magic,
        abi_major,
        abi_minor,
        header_size,
        record_count if header_record_count is None else header_record_count,
        epoch,
        epoch * 10,
        epoch * 1_000,
        slot,
    )
    for index, matrix in enumerate(matrices):
        if len(matrix) != 16:
            raise AssertionError("Synthetic transform must be a 4x4 matrix.")
        struct.pack_into(
            "<12f",
            payload,
            HEADER_SIZE + index * RECORD_STRIDE,
            *matrix[:12],
        )
    return bytes(payload)


def _identity_matrices(count: int) -> list[list[float]]:
    identity = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    return [identity.copy() for _ in range(count)]


def _world(scene, handles: list[int]) -> list[list[float]]:
    return [scene.get_node_world_transform_by_handle(handle) for handle in handles]


def _publish_expect_error(
    scene,
    stream,
    handles: list[int],
    slot: int,
    payload: bytes,
    expected_text: str,
) -> None:
    stream.debug_publish_slot(slot, payload)
    try:
        scene.consume_shared_transform_slot(stream, slot, handles)
    except RuntimeError as exc:
        assert expected_text.lower() in str(exc).lower(), str(exc)
    else:
        raise AssertionError(f"Expected shared transform error containing {expected_text!r}.")


def run(module_dir: Path, runtime_dir: Path, enable_debug: bool) -> None:
    sys.path.insert(0, str(module_dir.resolve()))
    renderer = importlib.import_module("DonutRenderPyNative")

    renderer.init(
        runtime_dir=str(runtime_dir.resolve()),
        backend="vulkan",
        device_index=0,
        enable_debug=enable_debug,
        enable_external_interop=True,
    )
    scene = None
    stream = None
    try:
        scene = renderer.create_scene()
        with tempfile.TemporaryDirectory(prefix="flora-fst-scene-") as temp_dir:
            scene_path = Path(temp_dir) / "dynamic_nodes.glb"
            _build_scene(scene_path)
            scene.load_scene(str(scene_path))

        handles = list(scene.get_node_handles(list(NODE_NAMES)))
        assert len(handles) == 2 and len(set(handles)) == 2
        stream = renderer.create_shared_transform_stream(
            record_count=len(handles),
            slot_count=3,
        )

        # Three slots over two rounds. For each non-symmetric transform, the
        # established CPU batch path is the golden result for the shared path.
        epoch = 0
        for _round in range(2):
            for slot in range(3):
                epoch += 1
                matrices = _matrices(epoch, slot)
                scene.update_node_transforms_batch(handles, matrices)
                expected_world = _world(scene, handles)

                scene.update_node_transforms_batch(
                    handles,
                    _identity_matrices(len(handles)),
                )
                stream.debug_publish_slot(
                    slot,
                    _payload(matrices, epoch=epoch, slot=slot),
                )
                token = scene.consume_shared_transform_slot(stream, slot, handles)

                assert token == {
                    "epoch": epoch,
                    "simulation_step": epoch * 10,
                    "timestamp_ns": epoch * 1_000,
                    "slot": slot,
                    "record_count": len(handles),
                }
                assert _world(scene, handles) == expected_world

        matrices = _matrices(epoch + 1, 0)
        _publish_expect_error(
            scene,
            stream,
            handles,
            0,
            _payload(matrices, epoch=epoch, slot=0),
            "stale",
        )
        _publish_expect_error(
            scene,
            stream,
            handles,
            1,
            _payload(
                matrices,
                epoch=epoch + 1,
                slot=1,
                header_record_count=1,
            ),
            "record_count",
        )
        _publish_expect_error(
            scene,
            stream,
            handles,
            2,
            _payload(matrices, epoch=epoch + 1, slot=2, magic=0),
            "magic",
        )

        nan_matrices = _matrices(epoch + 1, 0)
        nan_matrices[1][6] = math.nan
        _publish_expect_error(
            scene,
            stream,
            handles,
            0,
            _payload(nan_matrices, epoch=epoch + 1, slot=0),
            "finite",
        )
        _publish_expect_error(
            scene,
            stream,
            handles,
            1,
            _payload(matrices, epoch=epoch + 1, slot=1, abi_major=2),
            "abi",
        )
        _publish_expect_error(
            scene,
            stream,
            handles,
            2,
            _payload(matrices, epoch=epoch + 1, slot=2, header_size=48),
            "header_size",
        )

        # Invalid packets signal consumed but never advance the accepted epoch.
        recovery_epoch = epoch + 1
        recovery = _matrices(recovery_epoch, 0)
        scene.update_node_transforms_batch(handles, recovery)
        expected_recovery = _world(scene, handles)
        scene.update_node_transforms_batch(handles, _identity_matrices(len(handles)))
        stream.debug_publish_slot(
            0,
            _payload(recovery, epoch=recovery_epoch, slot=0),
        )
        recovery_token = scene.consume_shared_transform_slot(stream, 0, handles)
        assert recovery_token["epoch"] == recovery_epoch
        assert _world(scene, handles) == expected_recovery

        print(
            "PASS",
            {
                "slots": 3,
                "valid_updates": 7,
                "negative_packets": 6,
                "last_epoch": recovery_epoch,
                "debug": enable_debug,
            },
        )
    finally:
        if stream is not None:
            stream.close()
        renderer.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--enable-debug", action="store_true")
    args = parser.parse_args()
    run(args.module_dir, args.runtime_dir, args.enable_debug)


if __name__ == "__main__":
    main()
