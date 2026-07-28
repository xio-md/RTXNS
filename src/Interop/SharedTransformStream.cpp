#include "SharedTransformStream.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <nvrhi/nvrhi.h>
#include <nvrhi/vulkan.h>

#ifdef _WIN32
#include <vulkan/vulkan.h>
#endif

namespace rtxns::interop
{
    namespace
    {
        [[nodiscard]] uint64_t checked_add(uint64_t lhs, uint64_t rhs, const char* label)
        {
            if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
                throw std::overflow_error(std::string(label) + " exceeds uint64_t.");
            return lhs + rhs;
        }

        [[nodiscard]] uint64_t checked_multiply(uint64_t lhs, uint64_t rhs, const char* label)
        {
            if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
                throw std::overflow_error(std::string(label) + " exceeds uint64_t.");
            return lhs * rhs;
        }

        [[nodiscard]] uint64_t align_up(uint64_t value, uint64_t alignment)
        {
            const uint64_t remainder = value % alignment;
            return remainder == 0 ? value : checked_add(value, alignment - remainder, "aligned size");
        }

#ifdef _WIN32
        [[noreturn]] void throw_last_error(const char* operation)
        {
            throw std::system_error(
                static_cast<int>(GetLastError()),
                std::system_category(),
                operation);
        }

        [[nodiscard]] std::shared_ptr<OwnedWin32Handle> duplicate_handle(HANDLE source)
        {
            if (!source || source == INVALID_HANDLE_VALUE)
                throw std::runtime_error("Cannot duplicate an invalid Win32 handle.");

            HANDLE duplicate = nullptr;
            if (!DuplicateHandle(
                    GetCurrentProcess(),
                    source,
                    GetCurrentProcess(),
                    &duplicate,
                    0,
                    FALSE,
                    DUPLICATE_SAME_ACCESS))
            {
                throw_last_error("DuplicateHandle");
            }

            return std::make_shared<OwnedWin32Handle>(duplicate);
        }

        void check_vk(VkResult result, const char* operation)
        {
            if (result != VK_SUCCESS)
            {
                throw std::runtime_error(
                    std::string(operation) + " failed with VkResult " +
                    std::to_string(static_cast<int>(result)) + ".");
            }
        }
#endif
    }

    OwnedWin32Handle::OwnedWin32Handle(void* handle) noexcept
        : m_handle(handle)
    {
    }

    OwnedWin32Handle::~OwnedWin32Handle()
    {
        close();
    }

    OwnedWin32Handle::OwnedWin32Handle(OwnedWin32Handle&& other) noexcept
        : m_handle(std::exchange(other.m_handle, nullptr))
    {
    }

    OwnedWin32Handle& OwnedWin32Handle::operator=(OwnedWin32Handle&& other) noexcept
    {
        if (this != &other)
        {
            close();
            m_handle = std::exchange(other.m_handle, nullptr);
        }
        return *this;
    }

    uintptr_t OwnedWin32Handle::value() const noexcept
    {
        return reinterpret_cast<uintptr_t>(m_handle);
    }

    bool OwnedWin32Handle::closed() const noexcept
    {
        return m_handle == nullptr;
    }

    uintptr_t OwnedWin32Handle::detach() noexcept
    {
        return reinterpret_cast<uintptr_t>(std::exchange(m_handle, nullptr));
    }

    void OwnedWin32Handle::close() noexcept
    {
#ifdef _WIN32
        if (m_handle && m_handle != INVALID_HANDLE_VALUE)
            CloseHandle(static_cast<HANDLE>(m_handle));
#endif
        m_handle = nullptr;
    }

    class SharedTransformStream::Impl
    {
    public:
        struct Slot
        {
#ifdef _WIN32
            VkSemaphore ready = VK_NULL_HANDLE;
            VkSemaphore consumed = VK_NULL_HANDLE;
            HANDLE ready_export = nullptr;
            HANDLE consumed_export = nullptr;
#endif
        };

        ~Impl()
        {
            close_noexcept();
        }

        void initialize(
            std::shared_ptr<void> context_keepalive,
            nvrhi::IDevice* device,
            uint32_t record_count,
            uint32_t slot_count)
        {
#ifndef _WIN32
            (void)context_keepalive;
            (void)device;
            (void)record_count;
            (void)slot_count;
            throw std::runtime_error(
                "SharedTransformStream currently requires Windows OPAQUE_WIN32 external handles.");
#else
            if (!context_keepalive)
                throw std::invalid_argument("SharedTransformStream requires a live renderer context.");
            if (!device)
                throw std::invalid_argument("SharedTransformStream requires a valid NVRHI device.");
            if (device->getGraphicsAPI() != nvrhi::GraphicsAPI::VULKAN)
                throw std::invalid_argument("SharedTransformStream requires the Vulkan backend.");
            if (record_count == 0)
                throw std::invalid_argument("record_count must be greater than zero.");
            if (slot_count == 0)
                throw std::invalid_argument("slot_count must be greater than zero.");

            m_context_keepalive = std::move(context_keepalive);
            m_device = device;
            m_record_count = record_count;
            m_slot_count = slot_count;

            const uint64_t records_size = checked_multiply(
                record_count,
                SharedTransformStream::kRecordStride,
                "transform records size");
            m_slot_payload_size = checked_add(
                SharedTransformStream::kHeaderSize,
                records_size,
                "slot payload size");
            m_slot_stride = align_up(
                m_slot_payload_size,
                SharedTransformStream::kSlotAlignment);
            m_logical_size = checked_multiply(
                m_slot_stride,
                slot_count,
                "shared transform buffer size");

            m_vk_device = m_device->getNativeObject(nvrhi::ObjectTypes::VK_Device);
            m_vk_physical_device =
                m_device->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice);
            m_vk_instance = m_device->getNativeObject(nvrhi::ObjectTypes::VK_Instance);
            m_vulkan_device = m_device->getNativeObject(nvrhi::ObjectTypes::Nvrhi_VK_Device);
            if (!m_vk_device || !m_vk_physical_device || !m_vk_instance || !m_vulkan_device)
                throw std::runtime_error("Failed to acquire the native Vulkan objects from NVRHI.");

            load_vulkan_functions();
            query_device_uuid();

            nvrhi::BufferDesc shared_desc;
            shared_desc.byteSize = m_logical_size;
            shared_desc.debugName = "Flora SharedTransformStream";
            shared_desc.canHaveRawViews = true;
            shared_desc.initialState = nvrhi::ResourceStates::ShaderResource;
            shared_desc.keepInitialState = true;
            shared_desc.sharedResourceFlags = nvrhi::SharedResourceFlags::Shared;
            m_shared_buffer = m_device->createBuffer(shared_desc);
            if (!m_shared_buffer)
                throw std::runtime_error("Failed to create the exportable shared transform buffer.");

            const nvrhi::MemoryRequirements memory_requirements =
                m_device->getBufferMemoryRequirements(m_shared_buffer);
            m_allocation_size = memory_requirements.size;
            if (m_allocation_size < m_logical_size)
                throw std::runtime_error("The Vulkan buffer allocation is smaller than its logical size.");

            m_memory_export = m_shared_buffer->getNativeObject(nvrhi::ObjectTypes::SharedHandle);
            if (!m_memory_export || m_memory_export == INVALID_HANDLE_VALUE)
                throw std::runtime_error("NVRHI did not export a Win32 handle for the shared buffer.");

            nvrhi::BufferDesc readback_desc;
            readback_desc.byteSize = m_slot_payload_size;
            readback_desc.debugName = "Flora SharedTransformStream Readback";
            readback_desc.cpuAccess = nvrhi::CpuAccessMode::Read;
            readback_desc.initialState = nvrhi::ResourceStates::CopyDest;
            readback_desc.keepInitialState = true;
            m_readback_buffer = m_device->createBuffer(readback_desc);
            if (!m_readback_buffer)
                throw std::runtime_error("Failed to create the shared transform readback buffer.");

            m_slots.resize(slot_count);
            for (Slot& slot : m_slots)
            {
                slot.ready = create_exportable_binary_semaphore();
                slot.ready_export = export_semaphore(slot.ready);
                slot.consumed = create_exportable_binary_semaphore();
                slot.consumed_export = export_semaphore(slot.consumed);
            }

            // A newly-created slot is available to its first producer. Signal all
            // consumed semaphores in one real queue submission (not host signaling).
            nvrhi::CommandListHandle initialization = m_device->createCommandList();
            if (!initialization)
                throw std::runtime_error("Failed to create the semaphore initialization command list.");
            initialization->open();
            initialization->close();
            for (const Slot& slot : m_slots)
            {
                m_vulkan_device->queueSignalSemaphore(
                    nvrhi::CommandQueue::Graphics,
                    slot.consumed,
                    0);
            }
            m_device->executeCommandList(initialization, nvrhi::CommandQueue::Graphics);
            wait_for_graphics_queue();
#endif
        }

        [[nodiscard]] uint64_t slot_offset(uint32_t slot) const
        {
            validate_slot(slot);
            return checked_multiply(slot, m_slot_stride, "slot offset");
        }

        [[nodiscard]] std::shared_ptr<OwnedWin32Handle> duplicate_memory_handle() const
        {
            std::scoped_lock lock(m_mutex);
            validate_open();
#ifdef _WIN32
            return duplicate_handle(m_memory_export);
#else
            throw std::runtime_error("Win32 handles are unavailable on this platform.");
#endif
        }

        [[nodiscard]] std::shared_ptr<OwnedWin32Handle> duplicate_ready_handle(uint32_t slot) const
        {
            std::scoped_lock lock(m_mutex);
            validate_open();
            validate_slot(slot);
#ifdef _WIN32
            return duplicate_handle(m_slots[slot].ready_export);
#else
            throw std::runtime_error("Win32 handles are unavailable on this platform.");
#endif
        }

        [[nodiscard]] std::shared_ptr<OwnedWin32Handle> duplicate_consumed_handle(uint32_t slot) const
        {
            std::scoped_lock lock(m_mutex);
            validate_open();
            validate_slot(slot);
#ifdef _WIN32
            return duplicate_handle(m_slots[slot].consumed_export);
#else
            throw std::runtime_error("Win32 handles are unavailable on this platform.");
#endif
        }

        [[nodiscard]] std::vector<uint8_t> consume_slot(uint32_t slot)
        {
            std::scoped_lock lock(m_mutex);
            validate_open();
            validate_slot(slot);

#ifdef _WIN32
            nvrhi::CommandListHandle command_list = m_device->createCommandList();
            if (!command_list)
                throw std::runtime_error("Failed to create a debug-consume command list.");

            command_list->open();
            command_list->copyBuffer(
                m_readback_buffer,
                0,
                m_shared_buffer,
                slot_offset(slot),
                m_slot_payload_size);
            command_list->close();

            bool synchronization_registered = false;
            try
            {
                m_vulkan_device->queueWaitForSemaphore(
                    nvrhi::CommandQueue::Graphics,
                    m_slots[slot].ready,
                    0);
                synchronization_registered = true;
                m_vulkan_device->queueSignalSemaphore(
                    nvrhi::CommandQueue::Graphics,
                    m_slots[slot].consumed,
                    0);
                m_device->executeCommandList(command_list, nvrhi::CommandQueue::Graphics);
                wait_for_graphics_queue();
            }
            catch (...)
            {
                if (synchronization_registered)
                    m_poisoned = true;
                throw;
            }

            std::vector<uint8_t> result(static_cast<size_t>(m_slot_payload_size));
            void* mapped = m_device->mapBuffer(
                m_readback_buffer,
                nvrhi::CpuAccessMode::Read);
            if (!mapped)
                throw std::runtime_error("Failed to map the shared transform readback buffer.");

            std::memcpy(result.data(), mapped, result.size());
            m_device->unmapBuffer(m_readback_buffer);
            m_device->runGarbageCollection();
            return result;
#else
            throw std::runtime_error("SharedTransformStream is unavailable on this platform.");
#endif
        }

        void debug_publish_slot(uint32_t slot, const std::vector<uint8_t>& payload)
        {
            std::scoped_lock lock(m_mutex);
            validate_open();
            validate_slot(slot);
            if (payload.size() != m_slot_payload_size)
            {
                throw std::invalid_argument(
                    "debug_publish_slot payload length must equal slot_payload_size.");
            }

#ifdef _WIN32
            nvrhi::CommandListHandle command_list = m_device->createCommandList();
            if (!command_list)
                throw std::runtime_error("Failed to create a debug-publish command list.");

            command_list->open();
            command_list->writeBuffer(
                m_shared_buffer,
                payload.data(),
                payload.size(),
                slot_offset(slot));
            command_list->close();

            bool synchronization_registered = false;
            try
            {
                m_vulkan_device->queueWaitForSemaphore(
                    nvrhi::CommandQueue::Graphics,
                    m_slots[slot].consumed,
                    0);
                synchronization_registered = true;
                m_vulkan_device->queueSignalSemaphore(
                    nvrhi::CommandQueue::Graphics,
                    m_slots[slot].ready,
                    0);
                m_device->executeCommandList(command_list, nvrhi::CommandQueue::Graphics);
                wait_for_graphics_queue();
            }
            catch (...)
            {
                if (synchronization_registered)
                    m_poisoned = true;
                throw;
            }
#else
            (void)payload;
            throw std::runtime_error("SharedTransformStream is unavailable on this platform.");
#endif
        }

        void close()
        {
            std::scoped_lock lock(m_mutex);
            close_locked();
        }

        void close_noexcept() noexcept
        {
            try
            {
                std::scoped_lock lock(m_mutex);
                close_locked();
            }
            catch (...)
            {
                // Destructors must not cross the Python/C++ boundary with an exception.
            }
        }

        [[nodiscard]] bool closed() const noexcept
        {
            std::scoped_lock lock(m_mutex);
            return m_closed;
        }

        [[nodiscard]] bool poisoned() const noexcept
        {
            std::scoped_lock lock(m_mutex);
            return m_poisoned;
        }

        uint32_t m_record_count = 0;
        uint32_t m_slot_count = 0;
        uint64_t m_slot_payload_size = 0;
        uint64_t m_slot_stride = 0;
        uint64_t m_logical_size = 0;
        uint64_t m_allocation_size = 0;
        std::array<uint8_t, 16> m_device_uuid{};

    private:
        void validate_open() const
        {
            if (m_closed)
                throw std::runtime_error("SharedTransformStream is closed.");
            if (m_poisoned)
                throw std::runtime_error(
                    "SharedTransformStream synchronization is poisoned after a failed queue submission.");
        }

        void validate_slot(uint32_t slot) const
        {
            if (slot >= m_slot_count)
                throw std::out_of_range("SharedTransformStream slot index is out of range.");
        }

#ifdef _WIN32
        template <typename T>
        [[nodiscard]] T load_instance_function(const char* name)
        {
            auto function = reinterpret_cast<T>(m_vk_get_instance_proc_addr(m_vk_instance, name));
            if (!function)
                throw std::runtime_error(std::string("Vulkan function is unavailable: ") + name);
            return function;
        }

        template <typename T>
        [[nodiscard]] T load_device_function(const char* name)
        {
            auto function = reinterpret_cast<T>(m_vk_get_device_proc_addr(m_vk_device, name));
            if (!function)
                throw std::runtime_error(std::string("Vulkan function is unavailable: ") + name);
            return function;
        }

        void load_vulkan_functions()
        {
            m_vulkan_loader = LoadLibraryW(L"vulkan-1.dll");
            if (!m_vulkan_loader)
                throw_last_error("LoadLibraryW(vulkan-1.dll)");

            m_vk_get_instance_proc_addr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                GetProcAddress(m_vulkan_loader, "vkGetInstanceProcAddr"));
            if (!m_vk_get_instance_proc_addr)
                throw_last_error("GetProcAddress(vkGetInstanceProcAddr)");

            m_vk_get_device_proc_addr =
                load_instance_function<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
            m_vk_get_physical_device_properties2 =
                load_instance_function<PFN_vkGetPhysicalDeviceProperties2>(
                    "vkGetPhysicalDeviceProperties2");
            m_vk_create_semaphore =
                load_device_function<PFN_vkCreateSemaphore>("vkCreateSemaphore");
            m_vk_destroy_semaphore =
                load_device_function<PFN_vkDestroySemaphore>("vkDestroySemaphore");
            m_vk_get_semaphore_win32_handle =
                load_device_function<PFN_vkGetSemaphoreWin32HandleKHR>(
                    "vkGetSemaphoreWin32HandleKHR");
        }

        void query_device_uuid()
        {
            VkPhysicalDeviceIDProperties id_properties{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            properties.pNext = &id_properties;
            m_vk_get_physical_device_properties2(m_vk_physical_device, &properties);
            std::copy_n(
                id_properties.deviceUUID,
                m_device_uuid.size(),
                m_device_uuid.begin());
        }

        [[nodiscard]] VkSemaphore create_exportable_binary_semaphore()
        {
            VkExportSemaphoreCreateInfo export_info{
                VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
            export_info.handleTypes =
                VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

            VkSemaphoreCreateInfo create_info{
                VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            create_info.pNext = &export_info;

            VkSemaphore semaphore = VK_NULL_HANDLE;
            check_vk(
                m_vk_create_semaphore(
                    m_vk_device,
                    &create_info,
                    nullptr,
                    &semaphore),
                "vkCreateSemaphore");
            return semaphore;
        }

        [[nodiscard]] HANDLE export_semaphore(VkSemaphore semaphore)
        {
            VkSemaphoreGetWin32HandleInfoKHR handle_info{
                VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
            handle_info.semaphore = semaphore;
            handle_info.handleType =
                VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

            HANDLE handle = nullptr;
            check_vk(
                m_vk_get_semaphore_win32_handle(
                    m_vk_device,
                    &handle_info,
                    &handle),
                "vkGetSemaphoreWin32HandleKHR");
            if (!handle || handle == INVALID_HANDLE_VALUE)
                throw std::runtime_error("Vulkan exported an invalid semaphore handle.");
            return handle;
        }

        void wait_for_graphics_queue()
        {
            nvrhi::EventQueryHandle event = m_device->createEventQuery();
            if (!event)
                throw std::runtime_error("Failed to create a graphics-queue completion event.");
            m_device->setEventQuery(event, nvrhi::CommandQueue::Graphics);
            m_device->waitEventQuery(event);
        }
#endif

        void close_locked()
        {
            if (m_closed)
                return;
            m_closed = true;

#ifdef _WIN32
            if (m_device)
                m_device->waitForIdle();

            for (Slot& slot : m_slots)
            {
                if (slot.ready_export && slot.ready_export != INVALID_HANDLE_VALUE)
                    CloseHandle(slot.ready_export);
                slot.ready_export = nullptr;
                if (slot.consumed_export && slot.consumed_export != INVALID_HANDLE_VALUE)
                    CloseHandle(slot.consumed_export);
                slot.consumed_export = nullptr;

                if (m_vk_destroy_semaphore && m_vk_device && slot.ready)
                    m_vk_destroy_semaphore(m_vk_device, slot.ready, nullptr);
                slot.ready = VK_NULL_HANDLE;
                if (m_vk_destroy_semaphore && m_vk_device && slot.consumed)
                    m_vk_destroy_semaphore(m_vk_device, slot.consumed, nullptr);
                slot.consumed = VK_NULL_HANDLE;
            }

            // NVRHI owns the VkDeviceMemory, but its Vulkan buffer implementation
            // does not close this exported Win32 handle.
            if (m_memory_export && m_memory_export != INVALID_HANDLE_VALUE)
                CloseHandle(m_memory_export);
            m_memory_export = nullptr;
#endif

            m_readback_buffer = nullptr;
            m_shared_buffer = nullptr;
            m_slots.clear();
            m_device = nullptr;
            m_vulkan_device = nullptr;

#ifdef _WIN32
            m_vk_device = VK_NULL_HANDLE;
            m_vk_physical_device = VK_NULL_HANDLE;
            m_vk_instance = VK_NULL_HANDLE;
            if (m_vulkan_loader)
                FreeLibrary(m_vulkan_loader);
            m_vulkan_loader = nullptr;
#endif

            m_context_keepalive.reset();
        }

        mutable std::mutex m_mutex;
        std::shared_ptr<void> m_context_keepalive;
        nvrhi::IDevice* m_device = nullptr;
        nvrhi::vulkan::IDevice* m_vulkan_device = nullptr;
        nvrhi::BufferHandle m_shared_buffer;
        nvrhi::BufferHandle m_readback_buffer;
        std::vector<Slot> m_slots;
        bool m_closed = false;
        bool m_poisoned = false;

#ifdef _WIN32
        HMODULE m_vulkan_loader = nullptr;
        VkInstance m_vk_instance = VK_NULL_HANDLE;
        VkPhysicalDevice m_vk_physical_device = VK_NULL_HANDLE;
        VkDevice m_vk_device = VK_NULL_HANDLE;
        HANDLE m_memory_export = nullptr;
        PFN_vkGetInstanceProcAddr m_vk_get_instance_proc_addr = nullptr;
        PFN_vkGetDeviceProcAddr m_vk_get_device_proc_addr = nullptr;
        PFN_vkGetPhysicalDeviceProperties2 m_vk_get_physical_device_properties2 = nullptr;
        PFN_vkCreateSemaphore m_vk_create_semaphore = nullptr;
        PFN_vkDestroySemaphore m_vk_destroy_semaphore = nullptr;
        PFN_vkGetSemaphoreWin32HandleKHR m_vk_get_semaphore_win32_handle = nullptr;
#endif
    };

    SharedTransformStream::SharedTransformStream(
        std::shared_ptr<void> context_keepalive,
        nvrhi::IDevice* device,
        uint32_t record_count,
        uint32_t slot_count)
        : m_impl(std::make_unique<Impl>())
    {
        try
        {
            m_impl->initialize(
                std::move(context_keepalive),
                device,
                record_count,
                slot_count);
        }
        catch (...)
        {
            m_impl->close_noexcept();
            throw;
        }
    }

    SharedTransformStream::~SharedTransformStream() = default;

    uint32_t SharedTransformStream::record_count() const noexcept
    {
        return m_impl->m_record_count;
    }

    uint32_t SharedTransformStream::slot_count() const noexcept
    {
        return m_impl->m_slot_count;
    }

    uint64_t SharedTransformStream::slot_payload_size() const noexcept
    {
        return m_impl->m_slot_payload_size;
    }

    uint64_t SharedTransformStream::slot_stride() const noexcept
    {
        return m_impl->m_slot_stride;
    }

    uint64_t SharedTransformStream::logical_size() const noexcept
    {
        return m_impl->m_logical_size;
    }

    uint64_t SharedTransformStream::allocation_size() const noexcept
    {
        return m_impl->m_allocation_size;
    }

    uint64_t SharedTransformStream::memory_offset() const noexcept
    {
        return 0;
    }

    uint64_t SharedTransformStream::slot_offset(uint32_t slot) const
    {
        return m_impl->slot_offset(slot);
    }

    std::array<uint8_t, 16> SharedTransformStream::device_uuid() const noexcept
    {
        return m_impl->m_device_uuid;
    }

    bool SharedTransformStream::closed() const noexcept
    {
        return m_impl->closed();
    }

    bool SharedTransformStream::poisoned() const noexcept
    {
        return m_impl->poisoned();
    }

    std::shared_ptr<OwnedWin32Handle> SharedTransformStream::duplicate_memory_handle() const
    {
        return m_impl->duplicate_memory_handle();
    }

    std::shared_ptr<OwnedWin32Handle> SharedTransformStream::duplicate_ready_handle(
        uint32_t slot) const
    {
        return m_impl->duplicate_ready_handle(slot);
    }

    std::shared_ptr<OwnedWin32Handle> SharedTransformStream::duplicate_consumed_handle(
        uint32_t slot) const
    {
        return m_impl->duplicate_consumed_handle(slot);
    }

    std::vector<uint8_t> SharedTransformStream::consume_slot(uint32_t slot)
    {
        return m_impl->consume_slot(slot);
    }

    std::vector<uint8_t> SharedTransformStream::debug_consume_slot(uint32_t slot)
    {
        return consume_slot(slot);
    }

    std::vector<uint32_t> SharedTransformStream::debug_consume_slot_u32(uint32_t slot)
    {
        const std::vector<uint8_t> bytes = debug_consume_slot(slot);
        std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
        std::memcpy(words.data(), bytes.data(), bytes.size());
        return words;
    }

    void SharedTransformStream::debug_publish_slot(
        uint32_t slot,
        const std::vector<uint8_t>& payload)
    {
        m_impl->debug_publish_slot(slot, payload);
    }

    void SharedTransformStream::close()
    {
        m_impl->close();
    }
}
