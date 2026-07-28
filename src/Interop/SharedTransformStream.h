#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace nvrhi
{
    class IDevice;
}

namespace rtxns::interop
{
    class OwnedWin32Handle
    {
    public:
        explicit OwnedWin32Handle(void* handle) noexcept;
        ~OwnedWin32Handle();

        OwnedWin32Handle(const OwnedWin32Handle&) = delete;
        OwnedWin32Handle& operator=(const OwnedWin32Handle&) = delete;
        OwnedWin32Handle(OwnedWin32Handle&& other) noexcept;
        OwnedWin32Handle& operator=(OwnedWin32Handle&& other) noexcept;

        [[nodiscard]] uintptr_t value() const noexcept;
        [[nodiscard]] bool closed() const noexcept;
        [[nodiscard]] uintptr_t detach() noexcept;
        void close() noexcept;

    private:
        void* m_handle = nullptr;
    };

    class SharedTransformStream
    {
    public:
        static constexpr uint32_t kAbiMajor = 1;
        static constexpr uint32_t kAbiMinor = 0;
        static constexpr uint32_t kHeaderSize = 64;
        static constexpr uint32_t kRecordStride = 12 * sizeof(float);
        static constexpr uint32_t kSlotAlignment = 256;
        static constexpr uint32_t kHeaderMagic = 0x31545346; // "FST1" in little-endian memory.

        SharedTransformStream(
            std::shared_ptr<void> context_keepalive,
            nvrhi::IDevice* device,
            uint32_t record_count,
            uint32_t slot_count);
        ~SharedTransformStream();

        SharedTransformStream(const SharedTransformStream&) = delete;
        SharedTransformStream& operator=(const SharedTransformStream&) = delete;

        [[nodiscard]] uint32_t record_count() const noexcept;
        [[nodiscard]] uint32_t slot_count() const noexcept;
        [[nodiscard]] uint64_t slot_payload_size() const noexcept;
        [[nodiscard]] uint64_t slot_stride() const noexcept;
        [[nodiscard]] uint64_t logical_size() const noexcept;
        [[nodiscard]] uint64_t allocation_size() const noexcept;
        [[nodiscard]] uint64_t memory_offset() const noexcept;
        [[nodiscard]] uint64_t slot_offset(uint32_t slot) const;
        [[nodiscard]] std::array<uint8_t, 16> device_uuid() const noexcept;
        [[nodiscard]] bool closed() const noexcept;
        [[nodiscard]] bool poisoned() const noexcept;

        [[nodiscard]] std::shared_ptr<OwnedWin32Handle> duplicate_memory_handle() const;
        [[nodiscard]] std::shared_ptr<OwnedWin32Handle> duplicate_ready_handle(uint32_t slot) const;
        [[nodiscard]] std::shared_ptr<OwnedWin32Handle> duplicate_consumed_handle(uint32_t slot) const;

        // C++ consumer primitive used by Scene integration. This method is not
        // exposed to Python, so slot payloads never cross the Python boundary.
        [[nodiscard]] std::vector<uint8_t> consume_slot(uint32_t slot);

        // Waits on ready[slot], copies the complete logical slot to a CPU readback
        // buffer, and signals consumed[slot] in the same graphics-queue submission.
        [[nodiscard]] std::vector<uint8_t> debug_consume_slot(uint32_t slot);
        [[nodiscard]] std::vector<uint32_t> debug_consume_slot_u32(uint32_t slot);

        // Flora-only diagnostic producer. It follows the same protocol as CUDA:
        // wait consumed[slot], write one slot, then signal ready[slot].
        void debug_publish_slot(uint32_t slot, const std::vector<uint8_t>& payload);

        void close();

    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
