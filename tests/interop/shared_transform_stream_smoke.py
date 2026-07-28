from __future__ import annotations

import argparse
import ctypes
import importlib
import struct
import subprocess
import sys
from pathlib import Path


HEADER_MAGIC = 0x31545346
HEADER_SIZE = 64
RECORD_STRIDE = 48


def _payload(record_count: int, slot: int, epoch: int) -> bytes:
    payload = bytearray(HEADER_SIZE + record_count * RECORD_STRIDE)
    struct.pack_into(
        "<IHHIIQQQI",
        payload,
        0,
        HEADER_MAGIC,
        1,
        0,
        HEADER_SIZE,
        record_count,
        epoch,
        epoch * 10,
        epoch * 1_000,
        slot,
    )

    for record in range(record_count):
        base = float(epoch * 100 + slot * 10 + record)
        matrix = (
            1.0,
            0.0,
            0.0,
            base + 0.1,
            0.0,
            1.0,
            0.0,
            base + 0.2,
            0.0,
            0.0,
            1.0,
            base + 0.3,
        )
        struct.pack_into(
            "<12f",
            payload,
            HEADER_SIZE + record * RECORD_STRIDE,
            *matrix,
        )
    return bytes(payload)


def _cuda_uuid_hex() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    first = result.stdout.splitlines()[0].strip()
    if first.startswith("GPU-"):
        first = first[4:]
    return first.replace("-", "").lower()


def _close_descriptor_handles(descriptor: dict) -> None:
    descriptor["memory_handle"].close()
    for slot in descriptor["slots"]:
        slot["ready_handle"].close()
        slot["consumed_handle"].close()


def run(module_dir: Path, runtime_dir: Path, enable_debug: bool) -> None:
    sys.path.insert(0, str(module_dir.resolve()))
    renderer = importlib.import_module("DonutRenderPyNative")

    renderer.init(
        runtime_dir=str(runtime_dir.resolve()),
        backend="vulkan",
        device_index=0,
        enable_debug=enable_debug,
    )
    try:
        try:
            renderer.create_shared_transform_stream(record_count=2)
        except RuntimeError as exc:
            assert "disabled" in str(exc).lower()
        else:
            raise AssertionError("Interop must remain disabled by default.")
    finally:
        renderer.destroy()

    renderer.init(
        runtime_dir=str(runtime_dir.resolve()),
        backend="vulkan",
        device_index=0,
        enable_debug=enable_debug,
        enable_external_interop=True,
    )
    stream = None
    descriptor = None
    try:
        stream = renderer.create_shared_transform_stream(record_count=2)
        descriptor = stream.descriptor()

        assert descriptor["abi"] == "flora.shared_transform.v1"
        assert descriptor["slot_count"] == 3
        assert descriptor["header_size"] == HEADER_SIZE
        assert descriptor["record_stride"] == RECORD_STRIDE
        assert descriptor["slot_payload_size"] == HEADER_SIZE + 2 * RECORD_STRIDE
        assert descriptor["slot_stride"] % 256 == 0
        assert descriptor["allocation_size"] >= descriptor["logical_size"]
        assert len(descriptor["device_uuid"]) == 16
        assert len(descriptor["slots"]) == 3

        handles = [int(descriptor["memory_handle"])]
        for index, slot in enumerate(descriptor["slots"]):
            assert slot["index"] == index
            assert slot["offset"] == index * descriptor["slot_stride"]
            handles.extend(
                (int(slot["ready_handle"]), int(slot["consumed_handle"]))
            )
        assert all(handle != 0 for handle in handles)
        assert len(handles) == len(set(handles))

        second_descriptor = stream.descriptor()
        second_handles = [int(second_descriptor["memory_handle"])]
        for slot in second_descriptor["slots"]:
            second_handles.extend(
                (int(slot["ready_handle"]), int(slot["consumed_handle"]))
            )
        assert set(handles).isdisjoint(second_handles)
        detached_memory = second_descriptor["memory_handle"].detach()
        assert detached_memory != 0
        assert second_descriptor["memory_handle"].closed
        assert second_descriptor["memory_handle"].detach() == 0
        second_descriptor["memory_handle"].close()
        close_handle = ctypes.windll.kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_int
        assert close_handle(ctypes.c_void_p(detached_memory)) != 0
        _close_descriptor_handles(second_descriptor)

        cuda_uuid = _cuda_uuid_hex()
        if cuda_uuid is not None:
            assert descriptor["device_uuid_hex"] == cuda_uuid

        # Closing caller-owned duplicates must not affect Flora's original
        # exported handles or the underlying Vulkan resources.
        _close_descriptor_handles(descriptor)
        assert descriptor["memory_handle"].closed
        assert all(
            slot["ready_handle"].closed and slot["consumed_handle"].closed
            for slot in descriptor["slots"]
        )

        # Two rounds force every binary semaphore through signal/wait reuse.
        for epoch in range(1, 3):
            for slot in range(3):
                expected = _payload(2, slot, epoch)
                stream.debug_publish_slot(slot, expected)
                actual = stream.debug_consume_slot(slot)
                assert actual == expected

        print(
            "PASS",
            {
                "uuid": descriptor["device_uuid_hex"],
                "slot_count": descriptor["slot_count"],
                "slot_stride": descriptor["slot_stride"],
                "logical_size": descriptor["logical_size"],
                "allocation_size": descriptor["allocation_size"],
                "round_trips": 6,
            },
        )
    finally:
        if descriptor is not None:
            _close_descriptor_handles(descriptor)
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
