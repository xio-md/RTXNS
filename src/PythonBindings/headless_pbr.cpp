#include "headless_pbr.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <donut/app/Camera.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/math/math.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/Scene.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/View.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/ForwardShadingPass.h>
#include <donut/render/GBuffer.h>
#include <donut/render/GBufferFillPass.h>
#include <donut/render/GeometryPasses.h>
#include <nvrhi/utils.h>

#include "../Interop/SharedTransformStream.h"
#include "../RayTracedShadow/RayTracedShadowPass.h"
#include "../RayTracedShadow/SceneGeometryProvider.h"
#include "../RayTracedShadow/AccelerationStructure.h"
#include "../RayTracedShadow/OMMBaker.h"

namespace rtxns::python
{
    using donut::app::DeviceManager;
    using donut::engine::CommonRenderPasses;
    using donut::engine::DirectionalLight;
    using donut::engine::FramebufferFactory;
    using donut::engine::PlanarView;
    using donut::engine::Scene;
    using donut::engine::ShaderFactory;
    using donut::engine::TextureCache;
    using donut::render::ForwardShadingPass;
    using donut::render::GBufferFillPass;
    using donut::render::GBufferRenderTargets;
    using donut::render::MaterialIDPass;
    using donut::vfs::NativeFileSystem;
    using donut::vfs::RootFileSystem;
    using donut::math::radians;

    namespace
    {
        struct DeviceManagerDeleter
        {
            void operator()(DeviceManager* manager) const noexcept
            {
                delete manager;
            }
        };

        dm::float3 to_float3(const std::array<float, 3>& value)
        {
            return dm::float3(value[0], value[1], value[2]);
        }

        dm::float3 normalize_or_throw(dm::float3 value, const char* name)
        {
            const float length_sq = dm::dot(value, value);
            if (length_sq <= 1.0e-12f)
            {
                throw std::runtime_error(std::string(name) + " must be non-zero.");
            }
            return value / std::sqrt(length_sq);
        }

        std::filesystem::path resolve_framework_shader_dir(const std::filesystem::path& runtime_dir)
        {
            if (runtime_dir.empty())
            {
                return {};
            }

            const std::filesystem::path candidates[] = {
                runtime_dir / "bin" / "shaders" / "framework" / "spirv",
                runtime_dir / "shaders" / "framework" / "spirv",
                runtime_dir / "framework" / "spirv"
            };

            for (const auto& candidate : candidates)
            {
                if (std::filesystem::exists(candidate))
                {
                    return candidate;
                }
            }

            return {};
        }
    }

    class RendererContext : public std::enable_shared_from_this<RendererContext>
    {
    public:
        explicit RendererContext(const ContextInitOptions& options)
        {
            bool isD3D12 = (options.backend == "d3d12");
            if (!isD3D12 && options.backend != "vulkan")
            {
                throw std::runtime_error("The RTXNS Donut Python backend supports backend='vulkan' or 'd3d12'.");
            }
            if (options.enable_external_interop && isD3D12)
            {
                throw std::runtime_error(
                    "External CUDA interop currently requires backend='vulkan'.");
            }

            DeviceManager* raw_manager = DeviceManager::Create(
                isD3D12 ? nvrhi::GraphicsAPI::D3D12 : nvrhi::GraphicsAPI::VULKAN);
            if (!raw_manager)
            {
                throw std::runtime_error("Failed to create a device manager.");
            }

            m_device_manager.reset(raw_manager);

            donut::app::DeviceCreationParameters device_params;
            device_params.adapterIndex = options.device_index;
            device_params.enableDebugRuntime = options.enable_debug;
            device_params.enableNvrhiValidationLayer = options.enable_debug;
            device_params.backBufferWidth = 0;
            device_params.backBufferHeight = 0;
            device_params.startFullscreen = false;
            device_params.enableRayTracingExtensions = true;
            device_params.maxFramesInFlight = 1;
            device_params.swapChainFormat = nvrhi::Format::SRGBA8_UNORM;

            // Request OMM extension for Vulkan (D3D12 enables it automatically via DXR 1.2)
            if (!isD3D12)
            {
                device_params.optionalVulkanDeviceExtensions.push_back(
                    VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME);

                if (options.enable_external_interop)
                {
                    device_params.requiredVulkanDeviceExtensions.push_back(
                        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
                    device_params.requiredVulkanDeviceExtensions.push_back(
                        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
                    device_params.requiredVulkanDeviceExtensions.push_back(
                        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
                    device_params.requiredVulkanDeviceExtensions.push_back(
                        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
                }
            }

            if (!m_device_manager->CreateHeadlessDevice(device_params))
            {
                throw std::runtime_error("Failed to create a headless device.");
            }

            // Donut log defaults to MessageBox popups on errors — disable for headless
            donut::log::EnableOutputToMessageBox(false);

            // Detect OMM hardware support
            m_ommSupported = m_device_manager->GetDevice()->queryFeatureSupport(
                nvrhi::Feature::RayTracingOpacityMicromap);
            m_externalInteropEnabled = options.enable_external_interop;

            m_root_fs = std::make_shared<RootFileSystem>();

            const auto shader_dir = resolve_framework_shader_dir(options.runtime_dir);
            if (!shader_dir.empty())
            {
                m_root_fs->mount("/shaders/donut", shader_dir);
            }

            m_shader_factory = std::make_shared<ShaderFactory>(device(), m_root_fs, "/shaders");
            m_common_passes = std::make_shared<CommonRenderPasses>(device(), m_shader_factory);
        }

        ~RendererContext()
        {
            m_common_passes.reset();
            m_shader_factory.reset();
            m_root_fs.reset();

            if (m_device_manager)
            {
                if (m_device_manager->GetDevice())
                {
                    m_device_manager->GetDevice()->waitForIdle();
                }
                m_device_manager->Shutdown();
            }
        }

        [[nodiscard]] nvrhi::IDevice* device() const
        {
            return m_device_manager->GetDevice();
        }

        [[nodiscard]] const std::shared_ptr<ShaderFactory>& shader_factory() const
        {
            return m_shader_factory;
        }

        [[nodiscard]] const std::shared_ptr<CommonRenderPasses>& common_passes() const
        {
            return m_common_passes;
        }

        [[nodiscard]] bool isOmmSupported() const { return m_ommSupported; }
        [[nodiscard]] bool isExternalInteropEnabled() const
        {
            return m_externalInteropEnabled;
        }

    private:
        std::unique_ptr<DeviceManager, DeviceManagerDeleter> m_device_manager;
        std::shared_ptr<RootFileSystem> m_root_fs;
        std::shared_ptr<ShaderFactory> m_shader_factory;
        std::shared_ptr<CommonRenderPasses> m_common_passes;
        bool m_ommSupported = false;
        bool m_externalInteropEnabled = false;
    };

    // TODO(week2): Move to render_view_slot.h after stabilization.
    struct RenderViewSlot
    {
        CameraDesc desc;
        donut::app::FirstPersonCamera camera;
        PlanarView view;

        uint32_t width = 0;
        uint32_t height = 0;
        float z_near = 0.1f;
        float z_far = 1000.0f;

        std::shared_ptr<FramebufferFactory> framebufferFactory;
        nvrhi::TextureHandle colorTarget;
        nvrhi::TextureHandle depthTarget;
        nvrhi::StagingTextureHandle readbackTarget; // legacy, keep for compat

        nvrhi::TextureHandle shadowTarget;
        nvrhi::TextureHandle shadowBlurTemp;
        nvrhi::TextureHandle compositeOutput;
        nvrhi::TextureHandle litColorSRV;

        // A4 sensor products are allocated lazily on the first multimodal request.
        std::unique_ptr<GBufferRenderTargets> sensorGBuffer;
        std::shared_ptr<FramebufferFactory> sensorIdFramebuffer;
        nvrhi::TextureHandle sensorIdTarget;
        nvrhi::StagingTextureHandle sensorDepthReadback;
        nvrhi::StagingTextureHandle sensorNormalReadback;
        nvrhi::StagingTextureHandle sensorIdReadback;

        // Week 2: keep per-dispatch binding sets alive until command list finishes.
        std::vector<nvrhi::BindingSetHandle> frameBindingScratch;

        // Week 3+: readback ring with occupancy tracking.
        struct ReadbackRingSlot {
            nvrhi::StagingTextureHandle staging;
            uint64_t occupancyToken = 0;  // 0 = free; non-zero = batch token currently writing to this slot
        };
        // P0 fix: configurable ring depth with occupancy protection.
        // Default to 4; depth may change only when no batch is pending.
        static constexpr uint32_t kDefaultRingDepth = 4;
        std::vector<ReadbackRingSlot> readbackRing;
        uint32_t ringWriteIdx = 0;
        uint32_t ringDepth = kDefaultRingDepth;

        HeadlessPbrScene::FrameStats lastStats{};
    };

    class HeadlessPbrScene::Impl
    {
        struct MeshSensorLabel
        {
            uint32_t instance_id = 0;
            uint32_t semantic_id = 0;
        };

    public:
        explicit Impl(std::shared_ptr<RendererContext> context)
            : m_context(std::move(context))
        {
            m_native_fs = std::make_shared<NativeFileSystem>();
            m_texture_cache = std::make_shared<TextureCache>(m_context->device(), m_native_fs, nullptr);
            m_forward_pass = std::make_unique<ForwardShadingPass>(m_context->device(), m_context->common_passes());
            m_forward_pass->Init(*m_context->shader_factory(), ForwardShadingPass::CreateParameters{});

            m_sensorGBufferPass = std::make_unique<GBufferFillPass>(
                m_context->device(), m_context->common_passes());
            GBufferFillPass::CreateParameters gbufferParams;
            gbufferParams.enableDepthWrite = true;
            m_sensorGBufferPass->Init(*m_context->shader_factory(), gbufferParams);

            m_sensorIdPass = std::make_unique<MaterialIDPass>(
                m_context->device(), m_context->common_passes());
            GBufferFillPass::CreateParameters idParams;
            idParams.enableDepthWrite = false;
            m_sensorIdPass->Init(*m_context->shader_factory(), idParams);

            // Create default camera 0 slot and use legacy path directly.
            m_views.emplace_back();

            // Legacy path: set camera 0 directly (bypass set_camera_desc bridge).
            {
                CameraDesc desc;
                desc.position = {0.0f, 0.5f, 3.0f};
                desc.target = {0.0f, 0.0f, 0.0f};
                desc.up = {0.0f, 1.0f, 0.0f};
                desc.fov_degrees = 45.0f;
                desc.width = 512;
                desc.height = 512;
                desc.z_near = 0.1f;
                desc.z_far = 1000.0f;
                set_camera_desc(0, desc);
            }
        }

        ~Impl()
        {
            if (m_context && m_context->device())
            {
                m_context->device()->waitForIdle();
            }

            m_scene.reset();
            m_texture_cache.reset();
            m_forward_pass.reset();
            m_framebuffer_factory.reset();
            m_readback_target.Reset();
            m_depth_target.Reset();
            m_color_target.Reset();
        }

        void load_scene(const std::filesystem::path& scene_path)
        {
            if (!std::filesystem::exists(scene_path))
            {
                throw std::runtime_error("Scene file does not exist: " + scene_path.string());
            }

            auto* device = m_context->device();
            device->waitForIdle();
            device->runGarbageCollection();

            m_forward_pass->ResetBindingCache();
            m_sensorGBufferPass->ResetBindingCache();
            m_sensorIdPass->ResetBindingCache();
            for (auto& view : m_views)
            {
                view.frameBindingScratch.clear();
            }
            m_shadowAS = {};
            m_blasInputs.clear();
            m_shadowSceneResources = {};
            m_nodeHandles.clear();
            m_nodeHandleByName.clear();
            m_ambiguousNodeNames.clear();
            m_meshSensorLabels.clear();
            if (m_rtShadowPass)
            {
                m_rtShadowPass->setSceneResources(
                    device, rtxns::shadow::ShadowSceneResources{});
            }
            m_scene.reset();
            m_default_light.reset();
            m_texture_cache->Reset();
            m_ommCpuCache.clear();

            m_scene = std::make_unique<Scene>(
                device,
                *m_context->shader_factory(),
                m_native_fs,
                m_texture_cache,
                nullptr,
                nullptr);

            if (!m_scene->Load(scene_path))
            {
                m_scene.reset();
                throw std::runtime_error("Failed to load scene: " + scene_path.string());
            }

            m_texture_cache->ProcessRenderingThreadCommands(*m_context->common_passes(), 0.0f);
            m_texture_cache->LoadingFinished();

            if (m_default_light_requested || m_scene->GetSceneGraph()->GetLights().empty())
            {
                ensure_default_light_attached();
            }

            m_frame_index = 0;
            m_scene->RefreshSceneGraph(m_frame_index);
            rebuild_node_handle_table();
            rebuild_default_sensor_labels();
            m_shadowSceneResources = rtxns::shadow::SceneGeometryProvider::buildShadowSceneResources(
                device, *m_scene->GetSceneGraph());
            if (m_rtShadowPass && m_shadowSceneResources.instanceCount > 0)
            {
                m_rtShadowPass->setSceneResources(device, m_shadowSceneResources);
            }

            // Cache CPU-side index/UV + material data for alpha-tested meshes BEFORE
            // FinishedLoading() frees BufferGroup::indexData / texcoord1Data.
            // This cache is consumed by the first-frame OMM baking loop.
            m_ommCpuCache = rtxns::shadow::SceneGeometryProvider::cacheAlphaTestedMeshData(
                *m_scene->GetSceneGraph());

            // Pre-readback alpha textures while no render command list is open.
            // This avoids flushing the render command list in render_frame() and
            // avoids texture state tracking conflicts between command lists.
            {
                std::unordered_map<nvrhi::ITexture*, std::vector<float>*> texReadbackCache;
                for (auto& [meshPtr, entry] : m_ommCpuCache)
                {
                    if (!entry.hasAlphaTexture || !entry.alphaTexture || !entry.alphaTexture->texture)
                        continue;
                    auto texDesc = entry.alphaTexture->texture->getDesc();
                    auto fmt = texDesc.format;
                    if (fmt == nvrhi::Format::BC1_UNORM || fmt == nvrhi::Format::BC1_UNORM_SRGB ||
                        fmt == nvrhi::Format::BC2_UNORM || fmt == nvrhi::Format::BC2_UNORM_SRGB)
                        continue;

                    auto* texPtr = entry.alphaTexture->texture.Get();
                    auto cacheIt = texReadbackCache.find(texPtr);
                    if (cacheIt != texReadbackCache.end())
                    {
                        entry.alphaPixels = *cacheIt->second;
                        entry.texWidth = texDesc.width;
                        entry.texHeight = texDesc.height;
                        entry.alphaReadBack = true;
                    }
                    else
                    {
                        std::vector<float> pixels;
                        if (rtxns::shadow::OMMBaker::readAlphaTexture(
                                device, entry.alphaTexture->texture,
                                texDesc.width, texDesc.height, pixels))
                        {
                            entry.alphaPixels = pixels;
                            entry.texWidth = texDesc.width;
                            entry.texHeight = texDesc.height;
                            entry.alphaReadBack = true;
                            texReadbackCache[texPtr] = &entry.alphaPixels;
                        }
                    }
                }
            }

            m_scene->FinishedLoading(m_frame_index);
        }

        void set_camera_desc(uint32_t index, const CameraDesc& desc)
        {
            if (desc.width == 0 || desc.height == 0)
                throw std::runtime_error("Camera resolution must be positive.");
            if (desc.z_near <= 0.0f || desc.z_far <= desc.z_near)
                throw std::runtime_error("Camera clipping planes are invalid.");
            if (desc.fov_degrees <= 0.0f || desc.fov_degrees >= 179.0f)
                throw std::runtime_error("Camera FOV must be in the range (0, 179).");

            const auto pos = to_float3(desc.position);
            const auto tgt = to_float3(desc.target);
            const auto cam_up = normalize_or_throw(to_float3(desc.up), "up");

            if (dm::length(tgt - pos) <= 1.0e-6f)
                throw std::runtime_error("Camera target must differ from the camera position.");

            if (index >= m_views.size())
                throw std::out_of_range("Camera index out of range.");

            auto& slot = m_views[index];
            slot.desc = desc;
            resize_slot_targets(slot, desc.width, desc.height);

            slot.width = desc.width;
            slot.height = desc.height;
            slot.z_near = desc.z_near;
            slot.z_far = desc.z_far;

            slot.camera.LookAt(pos, tgt, cam_up);
            const float aspect = static_cast<float>(desc.width) / static_cast<float>(desc.height);
            slot.view.SetViewport(nvrhi::Viewport(0.0f, static_cast<float>(desc.width), 0.0f, static_cast<float>(desc.height), 0.0f, 1.0f));
            slot.view.SetMatrices(
                slot.camera.GetWorldToViewMatrix(),
                dm::perspProjD3DStyle(radians(desc.fov_degrees), aspect, desc.z_near, desc.z_far));
            slot.view.UpdateCache();

            // Keep legacy members in sync for camera 0 until Patch D refactor.
            if (index == 0) {
                m_width = desc.width;
                m_height = desc.height;
                m_z_near = desc.z_near;
                m_z_far = desc.z_far;
                m_framebuffer_factory = slot.framebufferFactory;
                m_color_target = slot.colorTarget;
                m_depth_target = slot.depthTarget;
                m_readback_target = slot.readbackTarget;
                m_shadowTarget = slot.shadowTarget;
                m_shadowBlurTemp = slot.shadowBlurTemp;
                m_compositeOutput = slot.compositeOutput;
                m_litColorSRV = slot.litColorSRV;
                m_view = slot.view;
                // FirstPersonCamera is not copyable; use LookAt.
                m_camera.LookAt(pos, tgt, cam_up);
            }
        }

        void set_camera(
            const std::array<float, 3>& position,
            const std::array<float, 3>& target,
            const std::array<float, 3>& up,
            float fov_degrees,
            uint32_t width,
            uint32_t height,
            float z_near,
            float z_far)
        {
            CameraDesc desc;
            desc.position = position;
            desc.target = target;
            desc.up = up;
            desc.fov_degrees = fov_degrees;
            desc.width = width;
            desc.height = height;
            desc.z_near = z_near;
            desc.z_far = z_far;
            set_camera_desc(0, desc);
        }

        void set_ambient(
            const std::array<float, 3>& top_rgb,
            const std::array<float, 3>& bottom_rgb)
        {
            m_ambient_top = to_float3(top_rgb);
            m_ambient_bottom = to_float3(bottom_rgb);
        }

        void set_default_light(
            const std::array<float, 3>& direction,
            const std::array<float, 3>& color,
            float irradiance)
        {
            if (irradiance <= 0.0f)
            {
                throw std::runtime_error("Light irradiance must be positive.");
            }

            m_default_light_requested = true;
            m_default_light_direction = normalize_or_throw(to_float3(direction), "direction");
            m_default_light_color = to_float3(color);
            m_default_light_irradiance = irradiance;

            if (m_scene)
            {
                ensure_default_light_attached();
            }
        }

        void update_node_transform(const std::string& name, const std::vector<float>& matrix_values)
        {
            const auto handles = get_node_handles({name});
            update_node_transforms_batch(handles, {matrix_values});
        }

        [[nodiscard]] uint32_t node_handle_count() const noexcept
        {
            return static_cast<uint32_t>(m_nodeHandles.size());
        }

        [[nodiscard]] std::vector<uint32_t> get_node_handles(
            const std::vector<std::string>& names) const
        {
            if (!m_scene || !m_scene->GetSceneGraph())
            {
                throw std::runtime_error("No scene has been loaded.");
            }

            std::vector<uint32_t> handles;
            handles.reserve(names.size());
            for (const auto& name : names)
            {
                if (m_ambiguousNodeNames.contains(name))
                {
                    throw std::runtime_error("Scene node name is ambiguous: " + name);
                }
                const auto found = m_nodeHandleByName.find(name);
                if (found == m_nodeHandleByName.end())
                {
                    throw std::runtime_error("Scene node not found: " + name);
                }
                handles.push_back(found->second);
            }
            return handles;
        }

        void update_node_transforms_batch(
            const std::vector<uint32_t>& handles,
            const std::vector<std::vector<float>>& matrices)
        {
            if (handles.size() != matrices.size())
            {
                throw std::invalid_argument(
                    "update_node_transforms_batch expects one 4x4 matrix per handle.");
            }

            std::unordered_set<uint32_t> uniqueHandles;
            std::vector<DecomposedNodeTransform> transforms;
            transforms.reserve(handles.size());
            for (size_t index = 0; index < handles.size(); ++index)
            {
                (void)node_from_handle(handles[index]);
                if (!uniqueHandles.insert(handles[index]).second)
                {
                    throw std::invalid_argument(
                        "update_node_transforms_batch does not accept duplicate handles.");
                }
                transforms.push_back(decompose_node_transform(matrices[index]));
            }

            for (size_t index = 0; index < handles.size(); ++index)
            {
                auto* node = node_from_handle(handles[index]);
                const auto& transform = transforms[index];
                node->SetTransform(
                    &transform.translation,
                    &transform.rotation,
                    &transform.scaling);
            }
        }

        [[nodiscard]] HeadlessPbrScene::SharedTransformConsumeToken
            consume_shared_transform_slot(
                const std::shared_ptr<rtxns::interop::SharedTransformStream>& stream,
                uint32_t slot,
                const std::vector<uint32_t>& handles)
        {
            if (!stream)
                throw std::invalid_argument("stream must not be None.");
            if (handles.size() != stream->record_count())
            {
                throw std::invalid_argument(
                    "consume_shared_transform_slot expects one node handle per "
                    "stream transform record.");
            }

            std::scoped_lock lock(m_sharedTransformMutex);

            // Reject bad caller arguments before consuming a ready slot.
            std::unordered_set<uint32_t> uniqueHandles;
            uniqueHandles.reserve(handles.size());
            for (uint32_t handle : handles)
            {
                (void)node_from_handle(handle);
                if (!uniqueHandles.insert(handle).second)
                {
                    throw std::invalid_argument(
                        "consume_shared_transform_slot does not accept duplicate handles.");
                }
            }

            const std::vector<uint8_t> payload = stream->consume_slot(slot);
            if (payload.size() != stream->slot_payload_size() ||
                payload.size() < rtxns::interop::SharedTransformStream::kHeaderSize)
            {
                throw std::runtime_error(
                    "Shared transform slot payload has an invalid byte length.");
            }

            const auto read_value = [&payload]<typename T>(size_t offset)
            {
                if (offset > payload.size() || sizeof(T) > payload.size() - offset)
                    throw std::runtime_error("Shared transform slot header is truncated.");
                T value{};
                std::memcpy(&value, payload.data() + offset, sizeof(T));
                return value;
            };

            const uint32_t magic = read_value.template operator()<uint32_t>(0);
            const uint16_t abiMajor = read_value.template operator()<uint16_t>(4);
            const uint16_t abiMinor = read_value.template operator()<uint16_t>(6);
            const uint32_t headerSize = read_value.template operator()<uint32_t>(8);
            const uint32_t recordCount = read_value.template operator()<uint32_t>(12);
            const uint64_t epoch = read_value.template operator()<uint64_t>(16);
            const uint64_t simulationStep = read_value.template operator()<uint64_t>(24);
            const uint64_t timestampNs = read_value.template operator()<uint64_t>(32);

            if (magic != rtxns::interop::SharedTransformStream::kHeaderMagic)
                throw std::runtime_error("Shared transform slot has invalid FST1 magic.");
            if (abiMajor != rtxns::interop::SharedTransformStream::kAbiMajor ||
                abiMinor != rtxns::interop::SharedTransformStream::kAbiMinor)
            {
                throw std::runtime_error(
                    "Shared transform slot has an unsupported FST1 ABI version.");
            }
            if (headerSize != rtxns::interop::SharedTransformStream::kHeaderSize)
            {
                throw std::runtime_error(
                    "Shared transform slot has an invalid FST1 header_size.");
            }
            if (recordCount != stream->record_count() ||
                recordCount != handles.size())
            {
                throw std::runtime_error(
                    "Shared transform slot record_count does not match the "
                    "stream and node handles.");
            }

            for (auto iterator = m_sharedTransformEpochs.begin();
                 iterator != m_sharedTransformEpochs.end();)
            {
                if (iterator->first.expired())
                    iterator = m_sharedTransformEpochs.erase(iterator);
                else
                    ++iterator;
            }

            const std::weak_ptr<rtxns::interop::SharedTransformStream> streamKey(stream);
            const auto previousEpoch = m_sharedTransformEpochs.find(streamKey);
            if (previousEpoch != m_sharedTransformEpochs.end() &&
                epoch <= previousEpoch->second)
            {
                throw std::runtime_error(
                    "Shared transform slot epoch is stale or non-monotonic.");
            }

            std::vector<std::vector<float>> matrices;
            matrices.reserve(recordCount);
            for (uint32_t recordIndex = 0; recordIndex < recordCount; ++recordIndex)
            {
                const size_t recordOffset =
                    rtxns::interop::SharedTransformStream::kHeaderSize +
                    static_cast<size_t>(recordIndex) *
                        rtxns::interop::SharedTransformStream::kRecordStride;

                float affineRowMajor[12]{};
                std::memcpy(
                    affineRowMajor,
                    payload.data() + recordOffset,
                    sizeof(affineRowMajor));
                if (!std::all_of(
                        std::begin(affineRowMajor),
                        std::end(affineRowMajor),
                        [](float value)
                        {
                            return std::isfinite(value);
                        }))
                {
                    throw std::runtime_error(
                        "Shared transform records must contain only finite float3x4 values.");
                }

                // The stream ABI and the established Python scene API both use
                // row-major flattened matrices. Preserve that input convention;
                // decompose_node_transform performs the Donut-specific transpose.
                std::vector<float> matrix(16, 0.0f);
                for (uint32_t row = 0; row < 3; ++row)
                {
                    for (uint32_t column = 0; column < 4; ++column)
                    {
                        matrix[row * 4 + column] =
                            affineRowMajor[row * 4 + column];
                    }
                }
                matrix[15] = 1.0f;
                matrices.push_back(std::move(matrix));
            }

            // Reuse the established CPU SceneGraph transform path. Scene::Refresh
            // and the existing CPU-side TLAS update remain unchanged.
            update_node_transforms_batch(handles, matrices);
            m_sharedTransformEpochs[streamKey] = epoch;

            HeadlessPbrScene::SharedTransformConsumeToken token;
            token.epoch = epoch;
            token.simulation_step = simulationStep;
            token.timestamp_ns = timestampNs;
            token.slot = slot;
            token.record_count = recordCount;
            return token;
        }

        [[nodiscard]] std::vector<float> get_node_world_transform(
            const std::string& name) const
        {
            const auto handles = get_node_handles({name});
            return get_node_world_transform_by_handle(handles.front());
        }

        [[nodiscard]] std::vector<float> get_node_world_transform_by_handle(
            uint32_t handle) const
        {
            return world_transform_values(node_from_handle(handle));
        }

        void set_node_labels(
            const std::vector<std::string>& nodeNames,
            const std::vector<uint32_t>& instanceIds,
            const std::vector<uint32_t>& semanticIds)
        {
            if (nodeNames.size() != instanceIds.size() ||
                nodeNames.size() != semanticIds.size())
            {
                throw std::invalid_argument(
                    "set_node_labels expects one instance and semantic id per node.");
            }
            if (!m_scene || !m_scene->GetSceneGraph())
            {
                throw std::runtime_error("No scene has been loaded.");
            }

            const auto handles = get_node_handles(nodeNames);
            std::unordered_map<const donut::engine::SceneGraphNode*, MeshSensorLabel> labelsByNode;
            labelsByNode.reserve(handles.size());
            std::unordered_set<uint32_t> uniqueInstanceIds;
            uniqueInstanceIds.reserve(instanceIds.size());
            for (size_t index = 0; index < handles.size(); ++index)
            {
                if (instanceIds[index] == 0)
                {
                    throw std::invalid_argument(
                        "Sensor instance id 0 is reserved for background pixels.");
                }
                if (!uniqueInstanceIds.insert(instanceIds[index]).second)
                {
                    throw std::invalid_argument(
                        "Sensor instance ids must be unique across labeled nodes.");
                }
                const auto* node = node_from_handle(handles[index]);
                if (!labelsByNode.emplace(
                        node,
                        MeshSensorLabel{instanceIds[index], semanticIds[index]}).second)
                {
                    throw std::invalid_argument(
                        "set_node_labels does not accept duplicate scene nodes.");
                }
            }

            const auto& meshInstances = m_scene->GetSceneGraph()->GetMeshInstances();
            std::vector<MeshSensorLabel> candidate(meshInstances.size());
            for (const auto& meshInstance : meshInstances)
            {
                if (!meshInstance || meshInstance->GetInstanceIndex() < 0)
                {
                    continue;
                }
                const auto rawIndex = static_cast<size_t>(meshInstance->GetInstanceIndex());
                if (rawIndex >= candidate.size())
                {
                    throw std::runtime_error("Scene mesh instance index is out of range.");
                }
                candidate[rawIndex] = MeshSensorLabel{
                    static_cast<uint32_t>(rawIndex + 1u),
                    0u,
                };
            }
            std::vector<bool> matched(handles.size(), false);
            std::unordered_map<const donut::engine::SceneGraphNode*, size_t> labelIndexByNode;
            for (size_t index = 0; index < handles.size(); ++index)
            {
                labelIndexByNode.emplace(node_from_handle(handles[index]), index);
            }

            for (const auto& meshInstance : meshInstances)
            {
                if (!meshInstance || meshInstance->GetInstanceIndex() < 0)
                {
                    continue;
                }
                const auto rawIndex = static_cast<size_t>(meshInstance->GetInstanceIndex());
                if (rawIndex >= candidate.size())
                {
                    throw std::runtime_error("Scene mesh instance index is out of range.");
                }
                for (auto* node = meshInstance->GetNode(); node; node = node->GetParent())
                {
                    const auto found = labelsByNode.find(node);
                    if (found == labelsByNode.end())
                    {
                        continue;
                    }
                    candidate[rawIndex] = found->second;
                    matched[labelIndexByNode.at(node)] = true;
                    break;
                }
            }

            for (size_t index = 0; index < matched.size(); ++index)
            {
                if (!matched[index])
                {
                    throw std::runtime_error(
                        "Sensor label node has no descendant mesh instance: " +
                        nodeNames[index]);
                }
            }
            m_meshSensorLabels = std::move(candidate);
        }

        [[nodiscard]] HeadlessPbrScene::SceneStats get_scene_stats() const
        {
            HeadlessPbrScene::SceneStats stats;
            if (!m_scene || !m_scene->GetSceneGraph())
            {
                return stats;
            }

            const auto& instances = m_scene->GetSceneGraph()->GetMeshInstances();
            stats.mesh_instances = static_cast<uint32_t>(instances.size());
            stats.shadow_instances = m_shadowSceneResources.instanceCount;

            std::unordered_set<const donut::engine::MeshInfo*> meshes;
            std::unordered_set<const donut::engine::MeshGeometry*> geometries;
            std::unordered_set<const donut::engine::Material*> materials;
            for (const auto& instance : instances)
            {
                if (!instance || !instance->GetMesh())
                {
                    continue;
                }
                const auto& mesh = instance->GetMesh();
                if (meshes.insert(mesh.get()).second)
                {
                    stats.unique_vertices += mesh->totalVertices;
                    stats.unique_indices += mesh->totalIndices;
                }
                for (const auto& geometry : mesh->geometries)
                {
                    if (!geometry)
                    {
                        continue;
                    }
                    geometries.insert(geometry.get());
                    if (geometry->material)
                    {
                        materials.insert(geometry->material.get());
                    }
                }
            }

            stats.unique_meshes = static_cast<uint32_t>(meshes.size());
            stats.unique_geometries = static_cast<uint32_t>(geometries.size());
            stats.unique_materials = static_cast<uint32_t>(materials.size());
            return stats;
        }

        // --- Ring depth configuration (P0) ---
        void set_ring_depth(uint32_t depth)
        {
            if (depth < 2) depth = 2;
            if (depth > 16) depth = 16;

            if (!m_pendingBatches.empty())
            {
                throw std::runtime_error(
                    "Cannot change the readback ring depth while batches are pending. "
                    "Read all submitted tokens first.");
            }

            m_defaultRingDepth = depth;
            if (m_views.empty())
                return;

            auto* device = m_context->device();
            device->waitForIdle();
            for (auto& view : m_views)
            {
                if (view.ringDepth == depth)
                    continue;

                view.ringDepth = depth;
                view.ringWriteIdx = 0;
                recreate_readback_ring(view, view.readbackTarget->getDesc());
            }
        }

        [[nodiscard]] uint32_t get_ring_depth() const noexcept
        {
            return m_views.empty() ? m_defaultRingDepth : m_views[0].ringDepth;
        }

        void enable_rt_shadows(bool enable)
        {
            m_rtShadowsEnabled = enable;
            // Always create the RT shadow pass (even when disabled) so that the
            // composite/tonemap path is used consistently for both no-shadow
            // and shadow frames. When disabled, the shadow target is cleared
            // to white (shadow=1) and composite acts as a pure tonemap pass.
            if (!m_rtShadowPass && m_context)
            {
                m_rtShadowPass = std::make_unique<rtxns::shadow::RayTracedShadowPass>();
                m_rtShadowPass->initialize(
                    m_context->device(),
                    m_context->shader_factory().get(),
                    m_width,
                    m_height);
            }
            if (enable && m_rtShadowPass && m_shadowSceneResources.instanceCount > 0)
            {
                m_rtShadowPass->setSceneResources(
                    m_context->device(),
                    m_shadowSceneResources);
            }
            if (!enable)
            {
                m_shadowAS = {};
                m_blasInputs.clear();
            }
        }

        void enable_shadow_blur(bool enable)
        {
            m_blurEnabled = enable;
        }

        void enable_omm(bool enable)
        {
            if (enable && !m_context->isOmmSupported())
            {
                std::cerr << "[RTXNS] WARNING: OMM requested but not supported by device. Ignoring." << std::endl;
                return;
            }
            m_ommEnabled = enable;
        }

        void set_shadow_samples(uint32_t n)
        {
            m_shadowSamples = std::max(1u, std::min(n, 64u));
        }

        void enable_omm_stress(bool enable)
        {
            m_ommStress = enable;
        }

        void set_omm_config(uint32_t subdiv, uint32_t format)
        {
            m_ommSubdiv = std::max(2u, std::min(subdiv, 12u));
            m_ommFormat = (format == 1) ? 1u : 2u; // 1=2-state, 2=4-state
        }

        bool load_omm_cache(const std::string& path)
        {
            std::ifstream f(path, std::ios::binary);
            if (!f) return false;

            struct Header { uint32_t magic, version, subdiv, format, numEntries; } hdr;
            f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
            if (!f || hdr.magic != 0x4F4D4D43 || hdr.version != 1) return false;
            if (hdr.subdiv != m_ommSubdiv || hdr.format != m_ommFormat)
            {
                std::cerr << "[RTXNS] OMM cache: subdiv/format mismatch, ignoring cache." << std::endl;
                return false;
            }

            m_ommBakeCache.clear();
            m_ommBakeCache.reserve(hdr.numEntries);
            for (uint32_t i = 0; i < hdr.numEntries; ++i)
            {
                CachedOmmBake entry;
                f.read(reinterpret_cast<char*>(&entry.blasIndex), sizeof(uint32_t));
                f.read(reinterpret_cast<char*>(&entry.indexCount), sizeof(uint32_t));
                f.read(reinterpret_cast<char*>(&entry.alphaCutoff), sizeof(float));

                auto readVec = [&](std::vector<uint8_t>& v) {
                    uint32_t sz; f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
                    v.resize(sz); f.read(reinterpret_cast<char*>(v.data()), sz);
                };
                readVec(entry.bakeResult.arrayData);
                readVec(entry.bakeResult.descArray);
                readVec(entry.bakeResult.indexBuffer);
                readVec(entry.bakeResult.descHistogramData);
                readVec(entry.bakeResult.indexHistogramData);

                f.read(reinterpret_cast<char*>(&entry.bakeResult.descCount), sizeof(uint32_t));
                f.read(reinterpret_cast<char*>(&entry.bakeResult.descHistogramCount), sizeof(uint32_t));
                f.read(reinterpret_cast<char*>(&entry.bakeResult.indexCount), sizeof(uint32_t));
                f.read(reinterpret_cast<char*>(&entry.bakeResult.indexHistogramCount), sizeof(uint32_t));
                f.read(reinterpret_cast<char*>(&entry.bakeResult.indexFormat), sizeof(uint32_t));

                if (!f) { m_ommBakeCache.clear(); return false; }
                m_ommBakeCache.push_back(std::move(entry));
            }

            m_ommCacheLoaded = true;
            std::cout << "[RTXNS] OMM cache: loaded " << m_ommBakeCache.size() << " entries from " << path << std::endl;
            return true;
        }

        bool save_omm_cache(const std::string& path)
        {
            if (m_ommBakeCache.empty()) return false;

            std::ofstream f(path, std::ios::binary);
            if (!f) return false;

            struct Header { uint32_t magic, version, subdiv, format, numEntries; };
            Header hdr = { 0x4F4D4D43, 1, m_ommSubdiv, m_ommFormat, (uint32_t)m_ommBakeCache.size() };
            f.write(reinterpret_cast<char*>(&hdr), sizeof(hdr));

            for (const auto& e : m_ommBakeCache)
            {
                f.write(reinterpret_cast<const char*>(&e.blasIndex), sizeof(uint32_t));
                f.write(reinterpret_cast<const char*>(&e.indexCount), sizeof(uint32_t));
                f.write(reinterpret_cast<const char*>(&e.alphaCutoff), sizeof(float));

                auto writeVec = [&](const std::vector<uint8_t>& v) {
                    uint32_t sz = (uint32_t)v.size();
                    f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
                    f.write(reinterpret_cast<const char*>(v.data()), sz);
                };
                writeVec(e.bakeResult.arrayData);
                writeVec(e.bakeResult.descArray);
                writeVec(e.bakeResult.indexBuffer);
                writeVec(e.bakeResult.descHistogramData);
                writeVec(e.bakeResult.indexHistogramData);

                f.write(reinterpret_cast<const char*>(&e.bakeResult.descCount), sizeof(uint32_t));
                f.write(reinterpret_cast<const char*>(&e.bakeResult.descHistogramCount), sizeof(uint32_t));
                f.write(reinterpret_cast<const char*>(&e.bakeResult.indexCount), sizeof(uint32_t));
                f.write(reinterpret_cast<const char*>(&e.bakeResult.indexHistogramCount), sizeof(uint32_t));
                f.write(reinterpret_cast<const char*>(&e.bakeResult.indexFormat), sizeof(uint32_t));
            }

            std::cout << "[RTXNS] OMM cache: saved " << m_ommBakeCache.size() << " entries to " << path << std::endl;
            return true;
        }

        [[nodiscard]] std::vector<uint8_t> render_frame()
        {
            if (!m_scene)
            {
                throw std::runtime_error("No scene has been loaded.");
            }

            using Clock = std::chrono::high_resolution_clock;
            auto tFrameStart = Clock::now();
            m_lastStats = {};

            auto* device = m_context->device();
            auto command_list = device->createCommandList();
            command_list->open();

            m_scene->Refresh(command_list, m_frame_index++);

            auto* framebuffer = m_framebuffer_factory->GetFramebuffer(m_view);
            nvrhi::utils::ClearColorAttachment(command_list, framebuffer, 0, nvrhi::Color(0.0f));
            command_list->clearDepthStencilTexture(m_depth_target, nvrhi::AllSubresources, true, 1.0f, false, 0);

            if (m_scene->GetSceneGraph()->GetLights().empty())
            {
                ensure_default_light_attached();
            }

            auto tRasterStart = Clock::now();

            ForwardShadingPass::Context pass_context;
            const std::vector<std::shared_ptr<donut::engine::LightProbe>> light_probes;
            m_forward_pass->PrepareLights(
                pass_context,
                command_list,
                m_scene->GetSceneGraph()->GetLights(),
                m_ambient_top,
                m_ambient_bottom,
                light_probes);

            donut::render::InstancedOpaqueDrawStrategy opaque_draws;
            donut::render::RenderCompositeView(
                command_list,
                &m_view,
                nullptr,
                *m_framebuffer_factory,
                m_scene->GetSceneGraph()->GetRootNode(),
                opaque_draws,
                *m_forward_pass,
                pass_context,
                "Opaque");

            donut::render::TransparentDrawStrategy transparent_draws;
            donut::render::RenderCompositeView(
                command_list,
                &m_view,
                nullptr,
                *m_framebuffer_factory,
                m_scene->GetSceneGraph()->GetRootNode(),
                transparent_draws,
                *m_forward_pass,
                pass_context,
                "Transparent");

            // ---- RT Shadow Pass ----
            bool useRTShadow = m_rtShadowsEnabled && m_rtShadowPass && m_rtShadowPass->isValid();
            m_lastStats.rt_shadows_enabled = useRTShadow;

            {
                auto tRasterEnd = Clock::now();
                m_lastStats.raster_ms = std::chrono::duration<double, std::milli>(tRasterEnd - tRasterStart).count();
            }

            if (useRTShadow)
            {
                // Build acceleration structures on first frame
                if (!m_shadowAS.built)
                {
                    auto tBlasStart = Clock::now();
                    m_blasInputs = rtxns::shadow::SceneGeometryProvider::extractFromScene(
                        *m_scene->GetSceneGraph());

                    // OMM stress mode: force all geometry to be non-opaque
                    if (m_ommStress)
                        for (auto& inp : m_blasInputs) inp.forceNonOpaque = true;

                    // ---- OMM Baking & Build (one-time, before BLAS build) ----
                    // Uses CPU data cached in load_scene() BEFORE FinishedLoading() freed it.
                    if (m_ommEnabled && m_context->isOmmSupported())
                    {
                        std::cout << "[RTXNS] OMM: " << (m_ommCacheLoaded ? "loading from cache" : "baking")
                                  << " alpha-tested geometry..." << std::endl;

                        rtxns::shadow::OMMBaker baker;
                        int diagTotal = 0, diagNoCache = 0, diagCompressed = 0, diagNoTex = 0, diagBakeable = 0;
                        std::vector<bool> cacheUsed(m_ommBakeCache.size(), false);

                        for (size_t bi = 0; bi < m_blasInputs.size(); ++bi)
                        {
                            auto& input = m_blasInputs[bi];
                            bool hasAlpha = false;
                            for (const auto& g : input.geometries)
                            {
                                if (g.isAlphaTested) { hasAlpha = true; break; }
                            }
                            if (!hasAlpha)
                                continue;

                            diagTotal++;
                            input.hasAlphaTestedGeometry = true;

                            // Compute total index count for this mesh (stable across runs)
                            uint32_t meshIdxCount = 0;
                            for (const auto& g : input.geometries)
                                meshIdxCount += g.indexCount;

                            // ---- Get bake result (from cache or by baking) ----
                            rtxns::shadow::OMMBakeResult bakeResult;

                            if (m_ommCacheLoaded)
                            {
                                // Find matching cache entry by indexCount
                                bool found = false;
                                for (size_t ci = 0; ci < m_ommBakeCache.size(); ++ci)
                                {
                                    if (!cacheUsed[ci] &&
                                        m_ommBakeCache[ci].indexCount == meshIdxCount &&
                                        m_ommBakeCache[ci].bakeResult.isValid())
                                    {
                                        bakeResult = m_ommBakeCache[ci].bakeResult;
                                        cacheUsed[ci] = true;
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found) continue;
                            }
                            else
                            {
                                // Look up pre-cached CPU data + material info for this mesh
                                auto cacheIt = m_ommCpuCache.find(input.meshInfo);
                                if (cacheIt == m_ommCpuCache.end() ||
                                    cacheIt->second.indexData.empty() ||
                                    cacheIt->second.texcoordData.empty())
                                {
                                    diagNoCache++;
                                    continue;
                                }

                                const auto& cache = cacheIt->second;

                                // Skip very large meshes to avoid excessive baking time
                                if (cache.indexData.size() > 20000)
                                {
                                    std::cout << "[RTXNS] OMM: mesh[" << bi << "] skipping large mesh ("
                                              << cache.indexData.size() << " indices)" << std::endl;
                                    m_ommBakeCache.push_back({(uint32_t)bi, meshIdxCount, 0.5f, {}});
                                    continue;
                                }

                                // Use pre-readback alpha data (populated in load_scene)
                                if (!cache.alphaReadBack || cache.alphaPixels.empty())
                                {
                                    diagNoTex++;
                                    m_ommBakeCache.push_back({(uint32_t)bi, meshIdxCount, 0.5f, {}});
                                    continue;
                                }

                                // Setup bake input from cached CPU data
                                rtxns::shadow::OMMBakeInput bakeIn;
                                bakeIn.alphaPixels = cache.alphaPixels;
                                bakeIn.texWidth = cache.texWidth;
                                bakeIn.texHeight = cache.texHeight;
                                bakeIn.alphaCutoff = cache.alphaCutoff;
                                bakeIn.subdivisionLevel = m_ommSubdiv;
                                bakeIn.format = static_cast<uint32_t>(m_ommFormat);
                                bakeIn.indexData = cache.indexData.data();
                                bakeIn.indexCount = static_cast<uint32_t>(cache.indexData.size());
                                bakeIn.indexStride = 4;
                                bakeIn.uvData = cache.texcoordData.data();
                                bakeIn.uvStride = sizeof(dm::float2);

                                diagBakeable++;
                                bakeResult = baker.bake(bakeIn);

                                // Store for later saving
                                m_ommBakeCache.push_back({(uint32_t)bi, meshIdxCount,
                                    cache.alphaCutoff, bakeResult});
                            }

                            if (bakeResult.isValid())
                            {
                                // Upload to GPU buffers (separate buffers for arrayData and descArray
                                // to avoid alignment issues with perOmmDescsOffset)
                                // Convert OMM index buffer to UINT_32 if needed (NVRHI only supports R16/R32)
                                if (bakeResult.indexFormat != 2) // 2 = UINT_32
                                {
                                    uint32_t idxCount = bakeResult.indexCount;
                                    std::vector<uint8_t> newIdx(idxCount * 4);
                                    for (uint32_t i = 0; i < idxCount; ++i)
                                    {
                                        uint32_t val = 0;
                                        if (bakeResult.indexFormat == 0) // UINT_8
                                            val = bakeResult.indexBuffer[i];
                                        else if (bakeResult.indexFormat == 1) // UINT_16
                                            val = reinterpret_cast<const uint16_t*>(bakeResult.indexBuffer.data())[i];
                                        std::memcpy(&newIdx[i * 4], &val, 4);
                                    }
                                    bakeResult.indexBuffer = std::move(newIdx);
                                    bakeResult.indexFormat = 2; // UINT_32
                                }


                                nvrhi::BufferDesc dataDesc;
                                dataDesc.byteSize = bakeResult.arrayData.size();
                                dataDesc.debugName = "OMMArrayData";
                                dataDesc.isAccelStructBuildInput = true;
                                auto ommDataBuf = device->createBuffer(dataDesc);

                                nvrhi::BufferDesc descBufDesc;
                                descBufDesc.byteSize = bakeResult.descArray.size();
                                descBufDesc.debugName = "OMMDescArray";
                                descBufDesc.isAccelStructBuildInput = true;
                                auto ommDescBuf = device->createBuffer(descBufDesc);

                                nvrhi::BufferDesc ibDesc;
                                ibDesc.byteSize = bakeResult.indexBuffer.size();
                                ibDesc.debugName = "OMMIndex";
                                ibDesc.isAccelStructBuildInput = true;
                                auto ommIbBuf = device->createBuffer(ibDesc);

                                if (ommDataBuf && ommDescBuf && ommIbBuf)
                                {
                                    nvrhi::rt::OpacityMicromapDesc ommDesc;
                                    ommDesc.flags = nvrhi::rt::OpacityMicromapBuildFlags::FastTrace;
                                    ommDesc.inputBuffer = ommDataBuf;
                                    ommDesc.perOmmDescs = ommDescBuf;
                                    ommDesc.perOmmDescsOffset = 0;

                                    // Convert OMM SDK OpacityMicromapUsageCount (8B: u32+u16+u16)
                                    // to NVRHI OpacityMicromapUsageCount (12B: u32+u32+enum)
                                    // reinterpret_cast would misalign fields and crash the driver.
                                    {
                                        #pragma pack(push, 1)
                                        struct OmmSdkUsageCount { uint32_t count; uint16_t subdiv; uint16_t format; };
                                        #pragma pack(pop)
                                        static_assert(sizeof(OmmSdkUsageCount) == 8, "OMM SDK usage count must be 8 bytes");

                                        auto* src = reinterpret_cast<const OmmSdkUsageCount*>(
                                            bakeResult.descHistogramData.data());
                                        uint32_t histCount = static_cast<uint32_t>(
                                            bakeResult.descHistogramData.size() / sizeof(OmmSdkUsageCount));
                                        ommDesc.counts.reserve(histCount);
                                        for (uint32_t i = 0; i < histCount; ++i)
                                        {
                                            nvrhi::rt::OpacityMicromapUsageCount c;
                                            c.count = src[i].count;
                                            c.subdivisionLevel = src[i].subdiv;
                                            c.format = static_cast<nvrhi::rt::OpacityMicromapFormat>(src[i].format);
                                            ommDesc.counts.push_back(c);
                                    }
                                }

                                    input.opacityMicromap = device->createOpacityMicromap(ommDesc);
                                    if (input.opacityMicromap)
                                    {
                                        // Write buffer data AND build OMM on the MAIN command list
                                        // (same command list as BLAS build and ray query)
                                        command_list->writeBuffer(ommDataBuf, bakeResult.arrayData.data(), bakeResult.arrayData.size());
                                        command_list->writeBuffer(ommDescBuf, bakeResult.descArray.data(), bakeResult.descArray.size());
                                        command_list->writeBuffer(ommIbBuf, bakeResult.indexBuffer.data(), bakeResult.indexBuffer.size());
                                        command_list->buildOpacityMicromap(input.opacityMicromap, ommDesc);

                                        input.ommIndexBuffer = ommIbBuf;
                                        // Convert index histogram from OMM SDK format to NVRHI format
                                        {
                                            #pragma pack(push, 1)
                                            struct OmmSdkUsageCount { uint32_t count; uint16_t subdiv; uint16_t format; };
                                            #pragma pack(pop)

                                            auto* src = reinterpret_cast<const OmmSdkUsageCount*>(
                                                bakeResult.indexHistogramData.data());
                                            uint32_t ihc = static_cast<uint32_t>(
                                                bakeResult.indexHistogramData.size() / sizeof(OmmSdkUsageCount));
                                            input.ommUsageCounts.reserve(ihc);
                                            for (uint32_t i = 0; i < ihc; ++i)
                                            {
                                                nvrhi::rt::OpacityMicromapUsageCount c;
                                                c.count = src[i].count;
                                                c.subdivisionLevel = src[i].subdiv;
                                                c.format = static_cast<nvrhi::rt::OpacityMicromapFormat>(src[i].format);
                                                input.ommUsageCounts.push_back(c);
                                            }
                                        }

                                        std::cout << "[RTXNS] OMM: mesh[" << bi << "] "
                                                  << bakeResult.descCount << " OMMs, "
                                                  << bakeResult.indexCount << " indices" << std::endl;
                                    }
                                    else
                                    {
                                        std::cerr << "[RTXNS] OMM: mesh[" << bi << "] createOpacityMicromap returned null!" << std::endl;
                                    }
                                }
                            }
                        }

                        // OMM baking diagnostic summary
                        {
                            int bakedCount = 0;
                            for (const auto& inp : m_blasInputs)
                                if (inp.hasAlphaTestedGeometry && inp.opacityMicromap) bakedCount++;

                            std::cout << "[RTXNS] OMM: diagnostic: " << diagTotal << " alpha meshes" << std::endl
                                      << "  no CPU cache: " << diagNoCache << std::endl
                                      << "  compressed tex: " << diagCompressed << std::endl
                                      << "  no alpha tex: " << diagNoTex << std::endl
                                      << "  bakeable: " << diagBakeable << std::endl
                                      << "  baked: " << bakedCount << std::endl;
                        }

                        device->waitForIdle();
                    }

                    // Build BLASes first, then we have handles for instance creation
                    if (!m_blasInputs.empty())
                    {
                        m_shadowAS.blasList = rtxns::shadow::AccelerationStructure::buildBLASes(
                            device, m_blasInputs);

                        auto instances = rtxns::shadow::AccelerationStructure::buildInstanceDescs(
                            *m_scene->GetSceneGraph(),
                            m_shadowAS.blasList,
                            m_blasInputs);
                        m_shadowAS.instances = instances;

                        // Build TLAS
                        if (!instances.empty())
                        {
                            auto cmdListTLAS = device->createCommandList();
                            cmdListTLAS->open();

                            nvrhi::rt::AccelStructDesc tlasDesc;
                            tlasDesc.setTopLevelMaxInstances(instances.size());
                            tlasDesc.setBuildFlags(
                                nvrhi::rt::AccelStructBuildFlags::PreferFastTrace |
                                nvrhi::rt::AccelStructBuildFlags::AllowUpdate);
                            tlasDesc.setDebugName("TLAS");

                            m_shadowAS.tlas = device->createAccelStruct(tlasDesc);
                            if (m_shadowAS.tlas)
                            {
                                cmdListTLAS->buildTopLevelAccelStruct(
                                    m_shadowAS.tlas,
                                    instances.data(),
                                    instances.size(),
                                    nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
                            }

                            cmdListTLAS->close();
                            device->executeCommandList(cmdListTLAS);
                            device->waitForIdle();

                            m_shadowAS.built = m_shadowAS.tlas != nullptr;
                        }
                    }
                    auto tBlasEnd = Clock::now();
                    m_lastStats.blas_build_ms = std::chrono::duration<double, std::milli>(tBlasEnd - tBlasStart).count();
                    m_lastStats.as_built_this_frame = true;
                } else {
                    // Per-frame TLAS update
                    auto tTlasStart = Clock::now();
                    auto instances = rtxns::shadow::AccelerationStructure::buildInstanceDescs(
                        *m_scene->GetSceneGraph(),
                        m_shadowAS.blasList,
                        m_blasInputs);
                    m_shadowAS.instances = instances;

                    rtxns::shadow::AccelerationStructure::updateTLAS(
                        command_list, m_shadowAS, instances);
                    auto tTlasEnd = Clock::now();
                    m_lastStats.tlas_build_ms = std::chrono::duration<double, std::milli>(tTlasEnd - tTlasStart).count();
                }

                if (m_shadowAS.tlas)
                {
                    // Use the explicit default light when one was requested.
                    // Scene-authored lights are neutralized in that case, but
                    // may still occupy index 0 in the graph; taking that
                    // first light made RT shadows disagree with raster light.
                    dm::float3 sunDir = dm::normalize(dm::float3(1.0f, 1.0f, 0.5f));  // towards the light by default
                    if (m_default_light_requested && m_default_light)
                    {
                        dm::float3 lightDir = dm::float3(
                            float(m_default_light->GetDirection().x),
                            float(m_default_light->GetDirection().y),
                            float(m_default_light->GetDirection().z));
                        sunDir = dm::normalize(-lightDir);
                    }
                    else if (!m_scene->GetSceneGraph()->GetLights().empty())
                    {
                        auto firstLight = m_scene->GetSceneGraph()->GetLights().front();
                        if (auto dirLight = std::dynamic_pointer_cast<DirectionalLight>(firstLight))
                        {
                            // Donut stores directional light travel direction; shadow rays need direction to light.
                            dm::float3 lightDir = dm::float3(
                                float(dirLight->GetDirection().x),
                                float(dirLight->GetDirection().y),
                                float(dirLight->GetDirection().z));
                            sunDir = dm::normalize(-lightDir);
                        }
                    }

                    rtxns::shadow::ShadowConstants shadowConstants;
                    shadowConstants.sunDirection = sunDir;
                    // Multi-sample sun jitter (~0.3° angular spread).
                    // Produces contact-hardening penumbra: narrow near occluder
                    // contacts, wider further away — same as solar disk angular diameter.
                    shadowConstants.sunJitter = 0.005f;
                    shadowConstants.invViewProj = m_view.GetInverseViewProjectionMatrix();
                    shadowConstants.invProj = m_view.GetInverseProjectionMatrix(false);
                    shadowConstants.invView = dm::affineToHomogeneous(m_view.GetInverseViewMatrix());
                    shadowConstants.projParams = dm::float2(m_z_near, m_z_far);
                    shadowConstants.imageSize = dm::float2(float(m_width), float(m_height));
                    shadowConstants.shadowEnabled = 1;
                    shadowConstants.shadowRayMask = 0xFFu;
                    shadowConstants.shadowSamples = m_shadowSamples;

                    // Zero the shadow target before dispatch
                    nvrhi::Color clearBlack(0.0f, 0.0f, 0.0f, 0.0f);
                    command_list->clearTextureFloat(m_shadowTarget,
                        nvrhi::AllSubresources, clearBlack);

                    auto tShadowStart = Clock::now();
                    m_rtShadowPass->renderShadow(
                        command_list,
                        m_shadowAS.tlas,
                        shadowConstants,
                        m_depth_target,
                        m_shadowTarget);

                    // Bilateral shadow blur: horizontal then vertical pass
                    if (m_blurEnabled)
                    {
                        shadowConstants.blurDirection = 0; // horizontal
                        m_rtShadowPass->blurShadow(
                            command_list, m_shadowTarget, m_shadowBlurTemp,
                            m_depth_target, shadowConstants);

                        shadowConstants.blurDirection = 1; // vertical
                        m_rtShadowPass->blurShadow(
                            command_list, m_shadowBlurTemp, m_shadowTarget,
                            m_depth_target, shadowConstants);
                    }

                    auto tShadowEnd = Clock::now();
                    m_lastStats.shadow_ray_ms = std::chrono::duration<double, std::milli>(tShadowEnd - tShadowStart).count();
                }
            }

            if (!m_rtShadowPass && m_context)
            {
                m_rtShadowPass = std::make_unique<rtxns::shadow::RayTracedShadowPass>();
                m_rtShadowPass->initialize(
                    m_context->device(),
                    m_context->shader_factory().get(),
                    m_width,
                    m_height);
            }

            if (!useRTShadow && m_shadowTarget)
            {
                command_list->setTextureState(m_shadowTarget, nvrhi::AllSubresources,
                    nvrhi::ResourceStates::UnorderedAccess);
                command_list->commitBarriers();
                nvrhi::Color clearWhite(1.0f, 1.0f, 1.0f, 1.0f);
                command_list->clearTextureFloat(m_shadowTarget,
                    nvrhi::AllSubresources, clearWhite);
            }

            if (m_rtShadowPass && m_rtShadowPass->isValid())
            {
                // Copy lit color to SRV-compatible texture
                command_list->setTextureState(m_color_target, nvrhi::AllSubresources,
                    nvrhi::ResourceStates::CopySource);
                command_list->setTextureState(m_litColorSRV, nvrhi::AllSubresources,
                    nvrhi::ResourceStates::CopyDest);
                command_list->commitBarriers();
                command_list->copyTexture(m_litColorSRV, nvrhi::TextureSlice(),
                    m_color_target, nvrhi::TextureSlice());
                command_list->setTextureState(m_litColorSRV, nvrhi::AllSubresources,
                    nvrhi::ResourceStates::ShaderResource);
                command_list->setTextureState(m_color_target, nvrhi::AllSubresources,
                    nvrhi::ResourceStates::RenderTarget);
                command_list->commitBarriers();

                // Composite: litColor * shadow -> tonemapped output.
                if (!m_shadowAS.tlas && m_shadowTarget)
                {
                    // No TLAS yet (or first frame building): use fully lit shadow
                    nvrhi::Color clearWhite(1.0f, 1.0f, 1.0f, 1.0f);
                    command_list->clearTextureFloat(m_shadowTarget,
                        nvrhi::AllSubresources, clearWhite);
                }

                auto tCompositeStart = Clock::now();

                m_rtShadowPass->compositeShadow(
                    command_list,
                    m_litColorSRV,
                    m_shadowTarget,
                    m_compositeOutput,
                    m_width,
                    m_height);
                auto tCompositeEnd = Clock::now();
                m_lastStats.composite_ms = std::chrono::duration<double, std::milli>(tCompositeEnd - tCompositeStart).count();

                // Copy composite output to readback staging
                command_list->setTextureState(m_compositeOutput, nvrhi::AllSubresources,
                    nvrhi::ResourceStates::CopySource);
                command_list->commitBarriers();
                command_list->copyTexture(m_readback_target, nvrhi::TextureSlice(),
                    m_compositeOutput, nvrhi::TextureSlice());
                command_list->setTextureState(m_compositeOutput, nvrhi::AllSubresources,
                    nvrhi::ResourceStates::UnorderedAccess);
                command_list->commitBarriers();
            }
            else
            {
                // Original flow: copy color → readback
                command_list->setTextureState(m_color_target, nvrhi::AllSubresources,
                    nvrhi::ResourceStates::CopySource);
                command_list->commitBarriers();
                command_list->copyTexture(m_readback_target, nvrhi::TextureSlice(),
                    m_color_target, nvrhi::TextureSlice());
                command_list->setTextureState(m_color_target, nvrhi::AllSubresources,
                    nvrhi::ResourceStates::RenderTarget);
                command_list->commitBarriers();
            }

            auto tReadbackStart = Clock::now();
            command_list->close();
            device->executeCommandList(command_list);
            device->waitForIdle();

            size_t row_pitch = 0;
            const auto* mapped = static_cast<const uint8_t*>(
                device->mapStagingTexture(m_readback_target, nvrhi::TextureSlice(),
                    nvrhi::CpuAccessMode::Read, &row_pitch));
            if (!mapped)
            {
                throw std::runtime_error("Failed to map the readback texture.");
            }

            const size_t row_bytes = static_cast<size_t>(m_width) * 4u;
            std::vector<uint8_t> pixels(static_cast<size_t>(m_width) * static_cast<size_t>(m_height) * 4u);

            for (uint32_t row = 0; row < m_height; ++row)
            {
                std::copy_n(
                    mapped + row_pitch * row,
                    row_bytes,
                    pixels.data() + row_bytes * row);
            }

            device->unmapStagingTexture(m_readback_target);

            auto tFrameEnd = Clock::now();
            m_lastStats.readback_ms = std::chrono::duration<double, std::milli>(tFrameEnd - tReadbackStart).count();
            m_lastStats.total_ms = std::chrono::duration<double, std::milli>(tFrameEnd - tFrameStart).count();
            return pixels;
        }

        [[nodiscard]] uint32_t width() const noexcept
        {
            return m_width;
        }

        [[nodiscard]] uint32_t height() const noexcept
        {
            return m_height;
        }

        [[nodiscard]] HeadlessPbrScene::FrameStats lastFrameStats() const noexcept
        {
            return m_lastStats;
        }

        // --- Multi-camera helpers (Week 1) ---

        uint32_t add_camera_slot(const CameraDesc& desc)
        {
            uint32_t index = static_cast<uint32_t>(m_views.size());
            m_views.emplace_back();
            m_views.back().ringDepth = m_defaultRingDepth;
            set_camera_desc(index, desc);
            return index;
        }

        uint32_t camera_count_impl() const noexcept
        {
            return static_cast<uint32_t>(m_views.size());
        }

        /// Sync legacy members from a camera slot, then record per-view work into cmdList.
        /// If use_ring is true, copy output to the slot's readback ring instead of legacy readbackTarget.
        void sync_and_record_view(nvrhi::ICommandList* cmdList, uint32_t camera_index, bool first_view, bool use_ring = false)
        {
            if (camera_index >= m_views.size())
                throw std::out_of_range("Camera index out of range.");

            auto& slot = m_views[camera_index];

            // Always sync legacy members from this camera's slot.
            {
                m_width = slot.width; m_height = slot.height;
                m_z_near = slot.z_near; m_z_far = slot.z_far;
                m_framebuffer_factory = slot.framebufferFactory;
                m_color_target = slot.colorTarget; m_depth_target = slot.depthTarget;
                m_readback_target = use_ring
                    ? slot.readbackRing[slot.ringWriteIdx % slot.ringDepth].staging
                    : slot.readbackTarget;
                m_shadowTarget = slot.shadowTarget; m_shadowBlurTemp = slot.shadowBlurTemp;
                m_compositeOutput = slot.compositeOutput; m_litColorSRV = slot.litColorSRV;
                m_view = slot.view;
                const auto pos = to_float3(slot.desc.position);
                const auto tgt = to_float3(slot.desc.target);
                const auto cam_up = normalize_or_throw(to_float3(slot.desc.up), "up");
                m_camera.LookAt(pos, tgt, cam_up);
            }

            // Record per-view work into the shared command list.
            using Clock = std::chrono::high_resolution_clock;

            auto* framebuffer = m_framebuffer_factory->GetFramebuffer(m_view);
            nvrhi::utils::ClearColorAttachment(cmdList, framebuffer, 0, nvrhi::Color(0.0f));
            cmdList->clearDepthStencilTexture(m_depth_target, nvrhi::AllSubresources, true, 1.0f, false, 0);

            // ---- Raster ----
            ForwardShadingPass::Context pass_context;
            const std::vector<std::shared_ptr<donut::engine::LightProbe>> light_probes;
            m_forward_pass->PrepareLights(pass_context, cmdList,
                m_scene->GetSceneGraph()->GetLights(), m_ambient_top, m_ambient_bottom, light_probes);

            donut::render::InstancedOpaqueDrawStrategy opaque_draws;
            donut::render::RenderCompositeView(cmdList, &m_view, nullptr, *m_framebuffer_factory,
                m_scene->GetSceneGraph()->GetRootNode(), opaque_draws, *m_forward_pass, pass_context, "Opaque");
            donut::render::TransparentDrawStrategy transparent_draws;
            donut::render::RenderCompositeView(cmdList, &m_view, nullptr, *m_framebuffer_factory,
                m_scene->GetSceneGraph()->GetRootNode(), transparent_draws, *m_forward_pass, pass_context, "Transparent");

            // ---- RT Shadow (if enabled) ----
            bool useRTShadow = m_rtShadowsEnabled && m_rtShadowPass && m_rtShadowPass->isValid();
            if (useRTShadow && m_shadowAS.tlas)
            {
                dm::float3 sunDir = dm::normalize(dm::float3(1.0f, 1.0f, 0.5f));
                if (m_default_light_requested && m_default_light) {
                    dm::float3 ld(float(m_default_light->GetDirection().x), float(m_default_light->GetDirection().y), float(m_default_light->GetDirection().z));
                    sunDir = dm::normalize(-ld);
                } else if (!m_scene->GetSceneGraph()->GetLights().empty()) {
                    auto firstLight = m_scene->GetSceneGraph()->GetLights().front();
                    if (auto dirLight = std::dynamic_pointer_cast<DirectionalLight>(firstLight)) {
                        dm::float3 ld(float(dirLight->GetDirection().x), float(dirLight->GetDirection().y), float(dirLight->GetDirection().z));
                        sunDir = dm::normalize(-ld);
                    }
                }
                rtxns::shadow::ShadowConstants sc;
                sc.sunDirection = sunDir; sc.sunJitter = 0.005f;
                sc.invViewProj = m_view.GetInverseViewProjectionMatrix();
                sc.invProj = m_view.GetInverseProjectionMatrix(false);
                sc.invView = dm::affineToHomogeneous(m_view.GetInverseViewMatrix());
                sc.projParams = dm::float2(m_z_near, m_z_far);
                sc.imageSize = dm::float2(float(m_width), float(m_height));
                sc.shadowEnabled = 1; sc.shadowRayMask = 0xFFu; sc.shadowSamples = m_shadowSamples;

                nvrhi::Color clearBlack(0,0,0,0);
                cmdList->clearTextureFloat(m_shadowTarget, nvrhi::AllSubresources, clearBlack);
                m_rtShadowPass->renderShadow(cmdList, m_shadowAS.tlas, sc, m_depth_target, m_shadowTarget);
                if (m_blurEnabled) {
                    sc.blurDirection = 0;
                    m_rtShadowPass->blurShadow(cmdList, m_shadowTarget, m_shadowBlurTemp, m_depth_target, sc);
                    sc.blurDirection = 1;
                    m_rtShadowPass->blurShadow(cmdList, m_shadowBlurTemp, m_shadowTarget, m_depth_target, sc);
                }
            }
            else if (m_shadowTarget) {
                cmdList->setTextureState(m_shadowTarget, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
                cmdList->commitBarriers();
                cmdList->clearTextureFloat(m_shadowTarget, nvrhi::AllSubresources, nvrhi::Color(1,1,1,1));
            }

            // ---- Composite + Copy to readback ----
            if (m_rtShadowPass && m_rtShadowPass->isValid())
            {
                cmdList->setTextureState(m_color_target, nvrhi::AllSubresources, nvrhi::ResourceStates::CopySource);
                cmdList->setTextureState(m_litColorSRV, nvrhi::AllSubresources, nvrhi::ResourceStates::CopyDest);
                cmdList->commitBarriers();
                cmdList->copyTexture(m_litColorSRV, nvrhi::TextureSlice(), m_color_target, nvrhi::TextureSlice());
                cmdList->setTextureState(m_litColorSRV, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
                cmdList->setTextureState(m_color_target, nvrhi::AllSubresources, nvrhi::ResourceStates::RenderTarget);
                cmdList->commitBarriers();

                m_rtShadowPass->compositeShadow(cmdList, m_litColorSRV, m_shadowTarget, m_compositeOutput, m_width, m_height);
                cmdList->setTextureState(m_compositeOutput, nvrhi::AllSubresources, nvrhi::ResourceStates::CopySource);
                cmdList->commitBarriers();
                cmdList->copyTexture(m_readback_target, nvrhi::TextureSlice(), m_compositeOutput, nvrhi::TextureSlice());
                cmdList->setTextureState(m_compositeOutput, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
            }
            else
            {
                cmdList->setTextureState(m_color_target, nvrhi::AllSubresources, nvrhi::ResourceStates::CopySource);
                cmdList->commitBarriers();
                cmdList->copyTexture(m_readback_target, nvrhi::TextureSlice(), m_color_target, nvrhi::TextureSlice());
                cmdList->setTextureState(m_color_target, nvrhi::AllSubresources, nvrhi::ResourceStates::RenderTarget);
            }
            cmdList->commitBarriers();
        }

        void record_sensor_view(
            nvrhi::ICommandList* cmdList,
            uint32_t cameraIndex,
            bool recordIds)
        {
            if (cameraIndex >= m_views.size())
            {
                throw std::out_of_range("Camera index out of range.");
            }
            auto& slot = m_views[cameraIndex];
            ensure_sensor_targets(slot);

            slot.sensorGBuffer->Clear(cmdList);
            GBufferFillPass::Context gbufferContext;
            donut::render::InstancedOpaqueDrawStrategy opaqueDraws;
            donut::render::RenderCompositeView(
                cmdList,
                &slot.view,
                &slot.view,
                *slot.sensorGBuffer->GBufferFramebuffer,
                m_scene->GetSceneGraph()->GetRootNode(),
                opaqueDraws,
                *m_sensorGBufferPass,
                gbufferContext,
                "SensorGBufferOpaque");
            donut::render::TransparentDrawStrategy transparentDraws;
            donut::render::RenderCompositeView(
                cmdList,
                &slot.view,
                &slot.view,
                *slot.sensorGBuffer->GBufferFramebuffer,
                m_scene->GetSceneGraph()->GetRootNode(),
                transparentDraws,
                *m_sensorGBufferPass,
                gbufferContext,
                "SensorGBufferTransparent");

            if (recordIds)
            {
                cmdList->clearTextureUInt(
                    slot.sensorIdTarget,
                    nvrhi::AllSubresources,
                    std::numeric_limits<uint32_t>::max());
                GBufferFillPass::Context idContext;
                donut::render::InstancedOpaqueDrawStrategy idOpaqueDraws;
                donut::render::RenderCompositeView(
                    cmdList,
                    &slot.view,
                    &slot.view,
                    *slot.sensorIdFramebuffer,
                    m_scene->GetSceneGraph()->GetRootNode(),
                    idOpaqueDraws,
                    *m_sensorIdPass,
                    idContext,
                    "SensorIdOpaque");
                donut::render::TransparentDrawStrategy idTransparentDraws;
                donut::render::RenderCompositeView(
                    cmdList,
                    &slot.view,
                    &slot.view,
                    *slot.sensorIdFramebuffer,
                    m_scene->GetSceneGraph()->GetRootNode(),
                    idTransparentDraws,
                    *m_sensorIdPass,
                    idContext,
                    "SensorIdTransparent");
            }

            cmdList->setTextureState(
                slot.sensorGBuffer->Depth,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::CopySource);
            cmdList->setTextureState(
                slot.sensorGBuffer->GBufferNormals,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::CopySource);
            if (recordIds)
            {
                cmdList->setTextureState(
                    slot.sensorIdTarget,
                    nvrhi::AllSubresources,
                    nvrhi::ResourceStates::CopySource);
            }
            cmdList->commitBarriers();
            cmdList->copyTexture(
                slot.sensorDepthReadback,
                nvrhi::TextureSlice(),
                slot.sensorGBuffer->Depth,
                nvrhi::TextureSlice());
            cmdList->copyTexture(
                slot.sensorNormalReadback,
                nvrhi::TextureSlice(),
                slot.sensorGBuffer->GBufferNormals,
                nvrhi::TextureSlice());
            if (recordIds)
            {
                cmdList->copyTexture(
                    slot.sensorIdReadback,
                    nvrhi::TextureSlice(),
                    slot.sensorIdTarget,
                    nvrhi::TextureSlice());
            }
            cmdList->setTextureState(
                slot.sensorGBuffer->Depth,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::DepthWrite);
            cmdList->setTextureState(
                slot.sensorGBuffer->GBufferNormals,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::RenderTarget);
            if (recordIds)
            {
                cmdList->setTextureState(
                    slot.sensorIdTarget,
                    nvrhi::AllSubresources,
                    nvrhi::ResourceStates::RenderTarget);
            }
            cmdList->commitBarriers();
        }

        // --- Shared AS build/update (Week 3 fix, used by all batch paths) ---

        void record_or_build_shadow_as(nvrhi::ICommandList* cmdList)
        {
            const auto recordStart = std::chrono::high_resolution_clock::now();
            m_lastStats.shadow_as_record_cpu_ms = 0.0;
            bool useRTShadow = m_rtShadowsEnabled && m_rtShadowPass && m_rtShadowPass->isValid();
            if (!useRTShadow) return;

            if (!m_shadowAS.built)
            {
                // First frame: build BLAS + TLAS from scratch
                auto* device = m_context->device();
                m_blasInputs = rtxns::shadow::SceneGeometryProvider::extractFromScene(*m_scene->GetSceneGraph());

                // OMM stress mode
                if (m_ommStress)
                    for (auto& inp : m_blasInputs) inp.forceNonOpaque = true;

                if (!m_blasInputs.empty())
                {
                    m_shadowAS.blasList = rtxns::shadow::AccelerationStructure::buildBLASes(device, m_blasInputs);
                    auto instances = rtxns::shadow::AccelerationStructure::buildInstanceDescs(
                        *m_scene->GetSceneGraph(), m_shadowAS.blasList, m_blasInputs);
                    m_shadowAS.instances = instances;

                    if (!instances.empty())
                    {
                        nvrhi::rt::AccelStructDesc tlasDesc;
                        tlasDesc.setTopLevelMaxInstances(instances.size());
                        tlasDesc.setBuildFlags(
                            nvrhi::rt::AccelStructBuildFlags::PreferFastTrace |
                            nvrhi::rt::AccelStructBuildFlags::AllowUpdate);
                        tlasDesc.setDebugName("TLAS");

                        m_shadowAS.tlas = device->createAccelStruct(tlasDesc);
                        if (m_shadowAS.tlas)
                        {
                            // Build TLAS on the main command list (batch-compatible)
                            cmdList->buildTopLevelAccelStruct(
                                m_shadowAS.tlas, instances.data(), instances.size(),
                                nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
                        }
                    }
                }

                m_shadowAS.built = (m_shadowAS.tlas != nullptr);
                m_lastStats.blas_build_ms = 0;
                m_lastStats.as_built_this_frame = true;

                // Note: OMM baking and cache logic deferred to render_frame() single-camera path.
                // Full integration of OMM into batch path requires async OMM bake or pre-baked cache.
            }
            else
            {
                // Subsequent frames: only update TLAS instances
                auto instances = rtxns::shadow::AccelerationStructure::buildInstanceDescs(
                    *m_scene->GetSceneGraph(), m_shadowAS.blasList, m_blasInputs);
                m_shadowAS.instances = instances;
                rtxns::shadow::AccelerationStructure::updateTLAS(cmdList, m_shadowAS, instances);
            }

            const auto recordEnd = std::chrono::high_resolution_clock::now();
            m_lastStats.shadow_as_record_cpu_ms =
                std::chrono::duration<double, std::milli>(recordEnd - recordStart).count();
        }

        // --- Async batch API (Week 3, production single-command-list path) ---

        uint64_t submit_frame_batch_impl(const std::vector<uint32_t>& indices)
        {
            if (indices.empty()) return 0;

            for (size_t i = 0; i < indices.size(); ++i)
            {
                const auto idx = indices[i];
                if (idx >= m_views.size())
                    throw std::out_of_range("Camera index out of range.");
                if (std::find(indices.begin(), indices.begin() + i, idx) != indices.begin() + i)
                    throw std::invalid_argument("Camera indices in a batch must be unique.");
            }
            if (!m_scene) throw std::runtime_error("No scene loaded.");
            auto* device = m_context->device();

            // Check ring occupancy
            for (auto idx : indices) {
                auto& slot = m_views[idx];
                if (slot.readbackRing[slot.ringWriteIdx % slot.ringDepth].occupancyToken != 0)
                    return 0;
            }

            if (!m_rtShadowPass && m_context) {
                m_rtShadowPass = std::make_unique<rtxns::shadow::RayTracedShadowPass>();
                m_rtShadowPass->initialize(device, m_context->shader_factory().get(), 0, 0);
            }

            auto cmdList = device->createCommandList();
            cmdList->open();

            m_lastStats = {};
            m_lastStats.rt_shadows_enabled = m_rtShadowsEnabled && m_rtShadowPass && m_rtShadowPass->isValid();
            const auto refreshStart = std::chrono::high_resolution_clock::now();
            m_scene->Refresh(cmdList, m_frame_index++);
            const auto refreshEnd = std::chrono::high_resolution_clock::now();
            m_lastStats.scene_refresh_cpu_ms =
                std::chrono::duration<double, std::milli>(refreshEnd - refreshStart).count();
            if (m_scene->GetSceneGraph()->GetLights().empty())
                ensure_default_light_attached();
            record_or_build_shadow_as(cmdList);

            for (uint32_t i = 0; i < static_cast<uint32_t>(indices.size()); ++i)
                sync_and_record_view(cmdList, indices[i], i == 0, true /*use_ring*/);

            cmdList->close();

            uint64_t composite_token = device->executeCommandList(cmdList);
            auto query = device->createEventQuery();
            device->setEventQuery(query, nvrhi::CommandQueue::Graphics);

            PendingBatch pending;
            pending.token = composite_token;
            pending.cameraIndices = indices;
            pending.commandList = cmdList;
            pending.query = query;

            for (auto idx : indices) {
                uint32_t rs = m_views[idx].ringWriteIdx % m_views[idx].ringDepth;
                pending.ringIndices.push_back(rs);
                m_views[idx].readbackRing[rs].occupancyToken = composite_token;
            }

            m_pendingBatches.push_back(pending);

            for (auto idx : indices)
                m_views[idx].ringWriteIdx = (m_views[idx].ringWriteIdx + 1) % m_views[idx].ringDepth;

            return composite_token;
        }

        bool is_batch_ready_impl(uint64_t token) const
        {
            if (!m_context) return false;
            auto* device = m_context->device();
            for (const auto& pb : m_pendingBatches) {
                if (pb.token == token)
                    return pb.query ? device->pollEventQuery(pb.query) : true;
            }
            return true;
        }

        std::vector<std::vector<uint8_t>> read_frame_batch_impl(uint64_t token)
        {
            auto* device = m_context->device();

            // Find and remove the pending batch
            PendingBatch found;
            bool matched = false;
            auto it = m_pendingBatches.begin();
            for (; it != m_pendingBatches.end(); ++it) {
                if (it->token == token) { found = *it; matched = true; break; }
            }
            if (!matched)
                throw std::runtime_error("Unknown batch token.");

            if (found.query) {
                device->waitEventQuery(found.query);
                device->resetEventQuery(found.query);
            }
            // NVRHI keeps submitted command buffers in an in-flight list until
            // the application explicitly collects them. The completed query
            // makes it safe to retire those buffers before the next batch.
            device->runGarbageCollection();
            m_pendingBatches.erase(it);

            // Readback using per-camera ring indices saved at submit time
            std::vector<std::vector<uint8_t>> outputs;
            outputs.reserve(found.cameraIndices.size());
            for (size_t i = 0; i < found.cameraIndices.size(); ++i) {
                auto idx = found.cameraIndices[i];
                uint32_t ringIdx = (i < found.ringIndices.size())
                    ? found.ringIndices[i] : 0u;
                auto& slot = m_views[idx];
                auto& ringSlot = slot.readbackRing[ringIdx].staging;
                if (!ringSlot)
                    throw std::runtime_error("Readback ring slot is null.");

                size_t row_pitch = 0;
                const auto* mapped = static_cast<const uint8_t*>(
                    device->mapStagingTexture(ringSlot, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &row_pitch));
                if (!mapped) throw std::runtime_error("Failed to map readback ring texture.");

                const size_t row_bytes = static_cast<size_t>(slot.width) * 4u;
                std::vector<uint8_t> pixels(row_bytes * slot.height);
                for (uint32_t row = 0; row < slot.height; ++row)
                    std::copy_n(mapped + row_pitch * row, row_bytes, pixels.data() + row_bytes * row);
                device->unmapStagingTexture(ringSlot);

                // P0: Release ring slot occupancy — now safe for next submit.
                slot.readbackRing[ringIdx].occupancyToken = 0;

                outputs.push_back(std::move(pixels));
            }
            return outputs;
        }

        /// Readback one view slot and return pixel bytes.
        std::vector<uint8_t> readback_slot(RenderViewSlot& slot)
        {
            auto* device = m_context->device();
            size_t row_pitch = 0;
            const auto* mapped = static_cast<const uint8_t*>(
                device->mapStagingTexture(slot.readbackTarget, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &row_pitch));
            if (!mapped) throw std::runtime_error("Failed to map readback texture.");

            const size_t row_bytes = static_cast<size_t>(slot.width) * 4u;
            std::vector<uint8_t> pixels(row_bytes * slot.height);
            for (uint32_t row = 0; row < slot.height; ++row)
                std::copy_n(mapped + row_pitch * row, row_bytes, pixels.data() + row_bytes * row);
            device->unmapStagingTexture(slot.readbackTarget);
            return pixels;
        }

        HeadlessPbrScene::SensorFrame read_sensor_slot(
            RenderViewSlot& slot,
            uint32_t productMask)
        {
            auto* device = m_context->device();
            HeadlessPbrScene::SensorFrame frame;
            frame.width = slot.width;
            frame.height = slot.height;
            const size_t pixelCount =
                static_cast<size_t>(slot.width) * static_cast<size_t>(slot.height);

            if ((productMask & HeadlessPbrScene::SensorDepth) != 0)
            {
                size_t rowPitch = 0;
                const auto* mapped = static_cast<const uint8_t*>(
                    device->mapStagingTexture(
                        slot.sensorDepthReadback,
                        nvrhi::TextureSlice(),
                        nvrhi::CpuAccessMode::Read,
                        &rowPitch));
                if (!mapped)
                {
                    throw std::runtime_error("Failed to map sensor depth texture.");
                }
                frame.depth_linear.resize(pixelCount);
                const float nearPlane = slot.z_near;
                const float farPlane = slot.z_far;
                for (uint32_t row = 0; row < slot.height; ++row)
                {
                    const auto* source = reinterpret_cast<const float*>(
                        mapped + rowPitch * row);
                    auto* destination = frame.depth_linear.data() +
                        static_cast<size_t>(row) * slot.width;
                    for (uint32_t column = 0; column < slot.width; ++column)
                    {
                        const float normalizedDepth = source[column];
                        destination[column] =
                            normalizedDepth >= 1.0f - 1.0e-7f
                            ? 0.0f
                            : nearPlane * farPlane /
                                (farPlane - normalizedDepth * (farPlane - nearPlane));
                    }
                }
                device->unmapStagingTexture(slot.sensorDepthReadback);
            }

            if ((productMask & HeadlessPbrScene::SensorNormal) != 0)
            {
                size_t rowPitch = 0;
                const auto* mapped = static_cast<const uint8_t*>(
                    device->mapStagingTexture(
                        slot.sensorNormalReadback,
                        nvrhi::TextureSlice(),
                        nvrhi::CpuAccessMode::Read,
                        &rowPitch));
                if (!mapped)
                {
                    throw std::runtime_error("Failed to map sensor normal texture.");
                }
                frame.normal_world.resize(pixelCount * 3u);
                for (uint32_t row = 0; row < slot.height; ++row)
                {
                    const auto* source = reinterpret_cast<const int16_t*>(
                        mapped + rowPitch * row);
                    for (uint32_t column = 0; column < slot.width; ++column)
                    {
                        const size_t sourceOffset = static_cast<size_t>(column) * 4u;
                        const size_t destinationOffset =
                            (static_cast<size_t>(row) * slot.width + column) * 3u;
                        dm::float3 normal(
                            std::max(-1.0f, source[sourceOffset] / 32767.0f),
                            std::max(-1.0f, source[sourceOffset + 1u] / 32767.0f),
                            std::max(-1.0f, source[sourceOffset + 2u] / 32767.0f));
                        const float lengthSquared = dm::dot(normal, normal);
                        if (lengthSquared > 1.0e-12f)
                        {
                            normal /= std::sqrt(lengthSquared);
                        }
                        frame.normal_world[destinationOffset] = normal.x;
                        frame.normal_world[destinationOffset + 1u] = normal.y;
                        frame.normal_world[destinationOffset + 2u] = normal.z;
                    }
                }
                device->unmapStagingTexture(slot.sensorNormalReadback);
            }

            if ((productMask & (HeadlessPbrScene::SensorInstance |
                    HeadlessPbrScene::SensorSemantic)) != 0)
            {
                size_t rowPitch = 0;
                const auto* mapped = static_cast<const uint8_t*>(
                    device->mapStagingTexture(
                        slot.sensorIdReadback,
                        nvrhi::TextureSlice(),
                        nvrhi::CpuAccessMode::Read,
                        &rowPitch));
                if (!mapped)
                {
                    throw std::runtime_error("Failed to map sensor id texture.");
                }
                if ((productMask & HeadlessPbrScene::SensorInstance) != 0)
                {
                    frame.instance.resize(pixelCount);
                }
                if ((productMask & HeadlessPbrScene::SensorSemantic) != 0)
                {
                    frame.semantic.resize(pixelCount);
                }
                for (uint32_t row = 0; row < slot.height; ++row)
                {
                    const auto* source = reinterpret_cast<const uint32_t*>(
                        mapped + rowPitch * row);
                    for (uint32_t column = 0; column < slot.width; ++column)
                    {
                        const uint32_t rawInstance =
                            source[static_cast<size_t>(column) * 4u + 1u];
                        MeshSensorLabel label;
                        if (rawInstance != std::numeric_limits<uint32_t>::max())
                        {
                            if (rawInstance >= m_meshSensorLabels.size())
                            {
                                device->unmapStagingTexture(slot.sensorIdReadback);
                                throw std::runtime_error(
                                    "Sensor pass returned an invalid mesh instance index.");
                            }
                            label = m_meshSensorLabels[rawInstance];
                        }
                        const size_t destinationOffset =
                            static_cast<size_t>(row) * slot.width + column;
                        if (!frame.instance.empty())
                        {
                            frame.instance[destinationOffset] = label.instance_id;
                        }
                        if (!frame.semantic.empty())
                        {
                            frame.semantic[destinationOffset] = label.semantic_id;
                        }
                    }
                }
                device->unmapStagingTexture(slot.sensorIdReadback);
            }
            return frame;
        }

        /// Single-command-list batch rendering (Week 2 Patch D).
        std::vector<std::vector<uint8_t>> render_frame_batch_v2(const std::vector<uint32_t>& indices)
        {
            if (indices.empty()) return {};
            for (auto idx : indices)
                if (idx >= m_views.size()) throw std::out_of_range("Camera index out of range.");

            if (!m_scene) throw std::runtime_error("No scene has been loaded.");
            auto* device = m_context->device();

            // --- Build RayTracedShadowPass lazily (once) ---
            if (!m_rtShadowPass && m_context) {
                m_rtShadowPass = std::make_unique<rtxns::shadow::RayTracedShadowPass>();
                m_rtShadowPass->initialize(device, m_context->shader_factory().get(), 0, 0);
            }

            // --- Create one command list for the entire batch ---
            auto cmdList = device->createCommandList();
            cmdList->open();

            // --- Shared work: Scene::Refresh + ensure light (once per batch) ---
            m_lastStats = {};
            const auto refreshStart = std::chrono::high_resolution_clock::now();
            m_scene->Refresh(cmdList, m_frame_index++);
            const auto refreshEnd = std::chrono::high_resolution_clock::now();
            m_lastStats.scene_refresh_cpu_ms =
                std::chrono::duration<double, std::milli>(refreshEnd - refreshStart).count();
            if (m_scene->GetSceneGraph()->GetLights().empty())
                ensure_default_light_attached();

            // --- RT shadow AS build/update (shared function) ---
            m_lastStats.rt_shadows_enabled = m_rtShadowsEnabled && m_rtShadowPass && m_rtShadowPass->isValid();
            record_or_build_shadow_as(cmdList);

            // --- Record per-camera work into the same command list ---
            for (auto idx : indices)
                sync_and_record_view(cmdList, idx, idx == indices.front());

            // --- Execute once ---
            cmdList->close();
            device->executeCommandList(cmdList);
            device->waitForIdle();

            // --- Readback all cameras ---
            std::vector<std::vector<uint8_t>> outputs;
            outputs.reserve(indices.size());
            for (auto idx : indices)
                outputs.push_back(readback_slot(m_views[idx]));

            return outputs;
        }

        std::vector<HeadlessPbrScene::SensorFrame> render_sensor_batch_impl(
            const std::vector<uint32_t>& indices,
            uint32_t productMask)
        {
            constexpr uint32_t validMask = HeadlessPbrScene::SensorAll;
            if (productMask == 0 || (productMask & ~validMask) != 0)
            {
                throw std::invalid_argument("Sensor product mask is empty or invalid.");
            }
            if (indices.empty())
            {
                return {};
            }
            for (size_t index = 0; index < indices.size(); ++index)
            {
                if (indices[index] >= m_views.size())
                {
                    throw std::out_of_range("Camera index out of range.");
                }
                if (std::find(indices.begin(), indices.begin() + index, indices[index]) !=
                    indices.begin() + index)
                {
                    throw std::invalid_argument(
                        "Camera indices in a sensor batch must be unique.");
                }
            }
            if (!m_scene)
            {
                throw std::runtime_error("No scene has been loaded.");
            }

            const bool wantsColor = (productMask & HeadlessPbrScene::SensorColor) != 0;
            const bool wantsGeometry =
                (productMask & (HeadlessPbrScene::SensorDepth |
                    HeadlessPbrScene::SensorNormal |
                    HeadlessPbrScene::SensorInstance |
                    HeadlessPbrScene::SensorSemantic)) != 0;
            const bool wantsIds =
                (productMask & (HeadlessPbrScene::SensorInstance |
                    HeadlessPbrScene::SensorSemantic)) != 0;

            if (!wantsGeometry)
            {
                auto colorFrames = render_frame_batch_v2(indices);
                std::vector<HeadlessPbrScene::SensorFrame> frames;
                frames.reserve(indices.size());
                for (size_t index = 0; index < indices.size(); ++index)
                {
                    HeadlessPbrScene::SensorFrame frame;
                    frame.width = m_views[indices[index]].width;
                    frame.height = m_views[indices[index]].height;
                    frame.color_rgba8 = std::move(colorFrames[index]);
                    frames.push_back(std::move(frame));
                }
                return frames;
            }

            for (const auto cameraIndex : indices)
            {
                ensure_sensor_targets(m_views[cameraIndex]);
            }

            auto* device = m_context->device();
            if (wantsColor && !m_rtShadowPass && m_context)
            {
                m_rtShadowPass = std::make_unique<rtxns::shadow::RayTracedShadowPass>();
                m_rtShadowPass->initialize(
                    device,
                    m_context->shader_factory().get(),
                    0,
                    0);
            }

            auto cmdList = device->createCommandList();
            cmdList->open();
            m_lastStats = {};
            m_lastStats.rt_shadows_enabled = wantsColor && m_rtShadowsEnabled &&
                m_rtShadowPass && m_rtShadowPass->isValid();
            const auto refreshStart = std::chrono::high_resolution_clock::now();
            m_scene->Refresh(cmdList, m_frame_index++);
            const auto refreshEnd = std::chrono::high_resolution_clock::now();
            m_lastStats.scene_refresh_cpu_ms =
                std::chrono::duration<double, std::milli>(refreshEnd - refreshStart).count();
            if (m_scene->GetSceneGraph()->GetLights().empty())
            {
                ensure_default_light_attached();
            }
            if (wantsColor)
            {
                record_or_build_shadow_as(cmdList);
            }

            const auto sensorRecordStart = std::chrono::high_resolution_clock::now();
            for (size_t index = 0; index < indices.size(); ++index)
            {
                if (wantsColor)
                {
                    sync_and_record_view(
                        cmdList,
                        indices[index],
                        index == 0,
                        false);
                }
                record_sensor_view(cmdList, indices[index], wantsIds);
            }
            const auto sensorRecordEnd = std::chrono::high_resolution_clock::now();
            m_lastStats.sensor_record_cpu_ms =
                std::chrono::duration<double, std::milli>(
                    sensorRecordEnd - sensorRecordStart).count();

            cmdList->close();
            device->executeCommandList(cmdList);
            device->waitForIdle();

            std::vector<HeadlessPbrScene::SensorFrame> frames;
            frames.reserve(indices.size());
            for (const auto cameraIndex : indices)
            {
                auto frame = read_sensor_slot(m_views[cameraIndex], productMask);
                if (wantsColor)
                {
                    frame.color_rgba8 = readback_slot(m_views[cameraIndex]);
                }
                frames.push_back(std::move(frame));
            }
            device->runGarbageCollection();
            return frames;
        }

        /// TODO(week2): Remove after v2 is validated.
        std::vector<uint8_t> render_frame_for_index(uint32_t camera_index)
        {
            if (camera_index >= m_views.size())
                throw std::out_of_range("Camera index out of range.");

            auto& slot = m_views[camera_index];
            m_width = slot.width; m_height = slot.height;
            m_z_near = slot.z_near; m_z_far = slot.z_far;
            m_framebuffer_factory = slot.framebufferFactory;
            m_color_target = slot.colorTarget; m_depth_target = slot.depthTarget;
            m_readback_target = slot.readbackTarget;
            m_shadowTarget = slot.shadowTarget; m_shadowBlurTemp = slot.shadowBlurTemp;
            m_compositeOutput = slot.compositeOutput; m_litColorSRV = slot.litColorSRV;
            m_view = slot.view;
            const auto pos = to_float3(slot.desc.position);
            const auto tgt = to_float3(slot.desc.target);
            const auto cam_up = normalize_or_throw(to_float3(slot.desc.up), "up");
            m_camera.LookAt(pos, tgt, cam_up);
            return render_frame();
        }

        /// Legacy batch (per-camera render_frame loop). Replace with v2 after validation.
        std::vector<std::vector<uint8_t>> render_frame_batch_impl(const std::vector<uint32_t>& indices)
        {
            if (indices.empty())
                return {};

            for (auto idx : indices) {
                if (idx >= m_views.size())
                    throw std::out_of_range("Camera index out of range for batch.");
            }

            std::vector<std::vector<uint8_t>> outputs;
            outputs.reserve(indices.size());
            for (auto idx : indices) {
                outputs.push_back(render_frame_for_index(idx));
            }
            return outputs;
        }

    private:
        struct DecomposedNodeTransform
        {
            dm::double3 translation = dm::double3::zero();
            dm::dquat rotation = dm::dquat::identity();
            dm::double3 scaling = dm::double3(1.0);
        };

        [[nodiscard]] static DecomposedNodeTransform decompose_node_transform(
            const std::vector<float>& matrixValues)
        {
            if (matrixValues.size() != 16)
            {
                throw std::invalid_argument(
                    "Node transforms must be 4x4 matrices flattened into 16 floats.");
            }
            if (!std::all_of(matrixValues.begin(), matrixValues.end(), [](float value)
                {
                    return std::isfinite(value);
                }))
            {
                throw std::invalid_argument("Node transforms must contain only finite values.");
            }

            dm::float4x4 donutMatrix{};
            for (int row = 0; row < 4; ++row)
            {
                for (int column = 0; column < 4; ++column)
                {
                    donutMatrix[row][column] = matrixValues[column * 4 + row];
                }
            }

            DecomposedNodeTransform result;
            const auto affine = dm::homogeneousToAffine(donutMatrix);
            dm::decomposeAffine(
                dm::daffine3(affine),
                &result.translation,
                &result.rotation,
                &result.scaling);
            return result;
        }

        [[nodiscard]] static std::vector<float> world_transform_values(
            const donut::engine::SceneGraphNode* node)
        {
            float affineValues[12]{};
            dm::affineToColumnMajor(node->GetLocalToWorldTransformFloat(), affineValues);
            std::vector<float> result(16, 0.0f);
            std::copy(std::begin(affineValues), std::end(affineValues), result.begin());
            result[15] = 1.0f;
            return result;
        }

        [[nodiscard]] donut::engine::SceneGraphNode* node_from_handle(uint32_t handle) const
        {
            if (!m_scene || !m_scene->GetSceneGraph())
            {
                throw std::runtime_error("No scene has been loaded.");
            }
            if (handle >= m_nodeHandles.size())
            {
                throw std::out_of_range("Scene node handle is out of range.");
            }
            return m_nodeHandles[handle];
        }

        void register_node_alias(const std::string& name, uint32_t handle)
        {
            if (name.empty() || m_ambiguousNodeNames.contains(name))
            {
                return;
            }
            const auto [found, inserted] = m_nodeHandleByName.emplace(name, handle);
            if (!inserted && found->second != handle)
            {
                m_nodeHandleByName.erase(found);
                m_ambiguousNodeNames.insert(name);
            }
        }

        void rebuild_node_handle_table()
        {
            m_nodeHandles.clear();
            m_nodeHandleByName.clear();
            m_ambiguousNodeNames.clear();
            if (!m_scene || !m_scene->GetSceneGraph())
            {
                return;
            }

            std::unordered_map<donut::engine::SceneGraphNode*, uint32_t> handleByNode;
            donut::engine::SceneGraphWalker walker(
                m_scene->GetSceneGraph()->GetRootNode().get());
            while (walker)
            {
                auto* node = walker.Get();
                const auto handle = static_cast<uint32_t>(m_nodeHandles.size());
                m_nodeHandles.push_back(node);
                handleByNode.emplace(node, handle);
                register_node_alias(node->GetName(), handle);
                walker.Next(true);
            }

            for (const auto& instance : m_scene->GetSceneGraph()->GetMeshInstances())
            {
                if (!instance)
                {
                    continue;
                }
                auto node = instance->GetNodeSharedPtr();
                const auto found = handleByNode.find(node.get());
                if (found != handleByNode.end())
                {
                    register_node_alias(instance->GetName(), found->second);
                }
            }
        }

        void rebuild_default_sensor_labels()
        {
            m_meshSensorLabels.clear();
            if (!m_scene || !m_scene->GetSceneGraph())
            {
                return;
            }

            const auto& meshInstances = m_scene->GetSceneGraph()->GetMeshInstances();
            m_meshSensorLabels.resize(meshInstances.size());
            for (const auto& meshInstance : meshInstances)
            {
                if (!meshInstance || meshInstance->GetInstanceIndex() < 0)
                {
                    continue;
                }
                const auto rawIndex = static_cast<size_t>(meshInstance->GetInstanceIndex());
                if (rawIndex >= m_meshSensorLabels.size())
                {
                    throw std::runtime_error("Scene mesh instance index is out of range.");
                }
                m_meshSensorLabels[rawIndex] = MeshSensorLabel{
                    static_cast<uint32_t>(rawIndex + 1u),
                    0u,
                };
            }
        }

        void ensure_default_light_attached()
        {
            if (!m_scene || !m_scene->GetSceneGraph() || !m_scene->GetSceneGraph()->GetRootNode())
            {
                return;
            }

            if (!m_default_light)
            {
                m_default_light = std::make_shared<DirectionalLight>();
                m_scene->GetSceneGraph()->AttachLeafNode(
                    m_scene->GetSceneGraph()->GetRootNode(),
                    m_default_light);
            }

            m_default_light->color = m_default_light_color;
            m_default_light->irradiance = m_default_light_irradiance;
            m_default_light->SetDirection(dm::double3(m_default_light_direction.x, m_default_light_direction.y, m_default_light_direction.z));

            // When a default light is explicitly requested, neutralize any scene-authored
            // directional lights so their (often very bright) irradiance doesn't blow out
            // the image.  We keep the default light as the sole directional light source.
            for (auto& light : m_scene->GetSceneGraph()->GetLights())
            {
                if (light == m_default_light)
                    continue;
                if (auto dirLight = std::dynamic_pointer_cast<DirectionalLight>(light))
                {
                    dirLight->irradiance = 0.0f;
                }
            }
        }

        void recreate_readback_ring(RenderViewSlot& slot, const nvrhi::TextureDesc& outputDesc)
        {
            auto* device = m_context->device();
            slot.readbackRing.clear();
            slot.readbackRing.resize(slot.ringDepth);
            for (auto& ring : slot.readbackRing)
            {
                ring.staging = device->createStagingTexture(outputDesc, nvrhi::CpuAccessMode::Read);
                ring.occupancyToken = 0;
            }
        }

        void ensure_sensor_targets(RenderViewSlot& slot)
        {
            if (slot.sensorGBuffer && slot.sensorIdTarget &&
                slot.sensorDepthReadback && slot.sensorNormalReadback &&
                slot.sensorIdReadback)
            {
                return;
            }
            if (slot.width == 0 || slot.height == 0)
            {
                throw std::runtime_error("Camera targets must be initialized first.");
            }

            auto* device = m_context->device();
            slot.sensorGBuffer = std::make_unique<GBufferRenderTargets>();
            slot.sensorGBuffer->Init(
                device,
                dm::uint2(slot.width, slot.height),
                1u,
                false,
                false);

            nvrhi::TextureDesc depthDesc;
            depthDesc.width = slot.width;
            depthDesc.height = slot.height;
            depthDesc.dimension = nvrhi::TextureDimension::Texture2D;
            depthDesc.debugName = "DonutRenderPy/SensorDepth";
            depthDesc.format = nvrhi::Format::D32;
            depthDesc.isRenderTarget = true;
            depthDesc.useClearValue = true;
            depthDesc.clearValue = nvrhi::Color(1.0f);
            depthDesc.initialState = nvrhi::ResourceStates::DepthWrite;
            depthDesc.keepInitialState = true;
            slot.sensorGBuffer->Depth = device->createTexture(depthDesc);
            slot.sensorGBuffer->GBufferFramebuffer->DepthTarget =
                slot.sensorGBuffer->Depth;

            nvrhi::TextureDesc idDesc;
            idDesc.width = slot.width;
            idDesc.height = slot.height;
            idDesc.dimension = nvrhi::TextureDimension::Texture2D;
            idDesc.debugName = "DonutRenderPy/SensorMaterialInstanceId";
            idDesc.format = nvrhi::Format::RGBA32_UINT;
            idDesc.isRenderTarget = true;
            idDesc.initialState = nvrhi::ResourceStates::RenderTarget;
            idDesc.keepInitialState = true;
            slot.sensorIdTarget = device->createTexture(idDesc);
            slot.sensorIdFramebuffer = std::make_shared<FramebufferFactory>(device);
            slot.sensorIdFramebuffer->RenderTargets = {slot.sensorIdTarget};
            slot.sensorIdFramebuffer->DepthTarget = slot.sensorGBuffer->Depth;

            nvrhi::TextureDesc stagingDesc;
            stagingDesc.width = slot.width;
            stagingDesc.height = slot.height;
            stagingDesc.dimension = nvrhi::TextureDimension::Texture2D;
            stagingDesc.debugName = "DonutRenderPy/SensorDepthReadback";
            stagingDesc.format = nvrhi::Format::D32;
            slot.sensorDepthReadback = device->createStagingTexture(
                stagingDesc, nvrhi::CpuAccessMode::Read);

            stagingDesc.debugName = "DonutRenderPy/SensorNormalReadback";
            stagingDesc.format = nvrhi::Format::RGBA16_SNORM;
            slot.sensorNormalReadback = device->createStagingTexture(
                stagingDesc, nvrhi::CpuAccessMode::Read);

            stagingDesc.debugName = "DonutRenderPy/SensorIdReadback";
            stagingDesc.format = nvrhi::Format::RGBA32_UINT;
            slot.sensorIdReadback = device->createStagingTexture(
                stagingDesc, nvrhi::CpuAccessMode::Read);
        }

        void resize_slot_targets(RenderViewSlot& slot, uint32_t width, uint32_t height)
        {
            if (width == slot.width && height == slot.height && slot.colorTarget && slot.depthTarget && slot.readbackTarget)
            {
                return;
            }

            auto* device = m_context->device();
            device->waitForIdle();

            slot.sensorGBuffer.reset();
            slot.sensorIdFramebuffer.reset();
            slot.sensorIdTarget.Reset();
            slot.sensorDepthReadback.Reset();
            slot.sensorNormalReadback.Reset();
            slot.sensorIdReadback.Reset();

            nvrhi::TextureDesc color_desc;
            color_desc.width = width;
            color_desc.height = height;
            color_desc.dimension = nvrhi::TextureDimension::Texture2D;
            color_desc.debugName = "DonutRenderPy/Color";
            color_desc.format = nvrhi::Format::RGBA16_FLOAT;
            color_desc.isRenderTarget = true;
            color_desc.initialState = nvrhi::ResourceStates::RenderTarget;
            color_desc.keepInitialState = true;

            nvrhi::TextureDesc output_desc;
            output_desc.width = width;
            output_desc.height = height;
            output_desc.dimension = nvrhi::TextureDimension::Texture2D;
            output_desc.debugName = "DonutRenderPy/Output";
            output_desc.format = nvrhi::Format::RGBA8_UNORM;

            nvrhi::TextureDesc depth_desc;
            depth_desc.width = width;
            depth_desc.height = height;
            depth_desc.dimension = nvrhi::TextureDimension::Texture2D;
            depth_desc.debugName = "DonutRenderPy/Depth";
            depth_desc.format = nvrhi::Format::D32;
            depth_desc.isRenderTarget = true;
            depth_desc.initialState = nvrhi::ResourceStates::DepthWrite;
            depth_desc.keepInitialState = true;

            slot.colorTarget = device->createTexture(color_desc);
            slot.depthTarget = device->createTexture(depth_desc);
            slot.readbackTarget = device->createStagingTexture(output_desc, nvrhi::CpuAccessMode::Read);

            recreate_readback_ring(slot, output_desc);

            // Shadow target: R8_UNORM, UAV-compatible
            nvrhi::TextureDesc shadow_desc;
            shadow_desc.width = width;
            shadow_desc.height = height;
            shadow_desc.dimension = nvrhi::TextureDimension::Texture2D;
            shadow_desc.debugName = "DonutRenderPy/Shadow";
            shadow_desc.format = nvrhi::Format::R8_UNORM;
            shadow_desc.isUAV = true;
            shadow_desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            shadow_desc.keepInitialState = true;
            slot.shadowTarget = device->createTexture(shadow_desc);

            // Temp texture for shadow blur ping-pong
            nvrhi::TextureDesc blur_desc = shadow_desc;
            blur_desc.debugName = "DonutRenderPy/ShadowBlurTemp";
            slot.shadowBlurTemp = device->createTexture(blur_desc);

            // SRV-compatible copy of the color target (RenderTarget-only textures can't be SRV)
            nvrhi::TextureDesc lit_srv_desc;
            lit_srv_desc.width = width;
            lit_srv_desc.height = height;
            lit_srv_desc.dimension = nvrhi::TextureDimension::Texture2D;
            lit_srv_desc.debugName = "DonutRenderPy/LitColorSRV";
            lit_srv_desc.format = nvrhi::Format::RGBA16_FLOAT;
            lit_srv_desc.initialState = nvrhi::ResourceStates::ShaderResource;
            lit_srv_desc.keepInitialState = true;
            slot.litColorSRV = device->createTexture(lit_srv_desc);

            // Composite output: tonemapped RGBA8_UNORM, UAV-compatible for compute write.
            nvrhi::TextureDesc composite_desc;
            composite_desc.width = width;
            composite_desc.height = height;
            composite_desc.dimension = nvrhi::TextureDimension::Texture2D;
            composite_desc.debugName = "DonutRenderPy/CompositeOutput";
            composite_desc.format = nvrhi::Format::RGBA8_UNORM;
            composite_desc.isUAV = true;
            composite_desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            composite_desc.keepInitialState = true;
            slot.compositeOutput = device->createTexture(composite_desc);

            slot.framebufferFactory = std::make_shared<FramebufferFactory>(device);
            slot.framebufferFactory->RenderTargets = {slot.colorTarget};
            slot.framebufferFactory->DepthTarget = slot.depthTarget;
        }

        // --- Legacy single-camera members (bridged to m_views[0], remove after Patch D) ---
        std::shared_ptr<RendererContext> m_context;
        std::shared_ptr<NativeFileSystem> m_native_fs;
        std::shared_ptr<TextureCache> m_texture_cache;
        std::unique_ptr<Scene> m_scene;
        std::unique_ptr<ForwardShadingPass> m_forward_pass;
        std::unique_ptr<GBufferFillPass> m_sensorGBufferPass;
        std::unique_ptr<MaterialIDPass> m_sensorIdPass;
        std::shared_ptr<FramebufferFactory> m_framebuffer_factory;
        nvrhi::TextureHandle m_color_target;
        nvrhi::TextureHandle m_depth_target;
        nvrhi::StagingTextureHandle m_readback_target;
        PlanarView m_view;
        donut::app::FirstPersonCamera m_camera;
        std::shared_ptr<DirectionalLight> m_default_light;

        // RT shadow members
        bool m_rtShadowsEnabled = false;
        bool m_blurEnabled = true; // toggle for A/B blur comparison
        bool m_ommEnabled = false;
        bool m_ommStress = false;      // force non-opaque mode for OMM A/B testing
        uint32_t m_shadowSamples = 4;  // rays per pixel
        uint32_t m_ommSubdiv = 5;      // OMM subdivision level
        uint32_t m_ommFormat = 2;      // 1=OC1_2_State, 2=OC1_4_State
        std::unique_ptr<rtxns::shadow::RayTracedShadowPass> m_rtShadowPass;
        nvrhi::TextureHandle m_shadowTarget;
        nvrhi::TextureHandle m_shadowBlurTemp;
        nvrhi::TextureHandle m_compositeOutput;
        nvrhi::TextureHandle m_litColorSRV;
        rtxns::shadow::ShadowAccelStructures m_shadowAS;
        rtxns::shadow::ShadowSceneResources m_shadowSceneResources;  // alpha-test metadata
        rtxns::shadow::OMMCpuCache m_ommCpuCache; // CPU data for OMM baking (captured pre-FinishedLoading)
        std::vector<rtxns::shadow::MeshBLASInput> m_blasInputs;

        // OMM bake cache: stores bake results to avoid re-baking on every test run
        struct CachedOmmBake {
            uint32_t blasIndex;      // index into m_blasInputs
            uint32_t indexCount;     // for verification
            float    alphaCutoff;
            rtxns::shadow::OMMBakeResult bakeResult;
        };
        std::vector<CachedOmmBake> m_ommBakeCache;  // loaded from disk
        bool m_ommCacheLoaded = false;

    private:
        // --- Legacy lighting members ---
        dm::float3 m_ambient_top = dm::float3(0.03f, 0.04f, 0.06f);
        dm::float3 m_ambient_bottom = dm::float3(0.01f, 0.01f, 0.01f);

        bool m_default_light_requested = false;
        dm::float3 m_default_light_direction = normalize_or_throw(dm::float3(-0.4f, -1.0f, -0.6f), "default light direction");
        dm::float3 m_default_light_color = dm::float3(1.0f, 1.0f, 1.0f);
        float m_default_light_irradiance = 2.0f;

        uint32_t m_width = 0;
        uint32_t m_height = 0;
        float m_z_near = 0.1f;
        float m_z_far = 1000.0f;
        uint32_t m_frame_index = 0;
        uint32_t m_defaultRingDepth = RenderViewSlot::kDefaultRingDepth;

        HeadlessPbrScene::FrameStats m_lastStats{};

        // --- Multi-camera slots (Week 1) ---
        std::vector<RenderViewSlot> m_views;

        // A3: scene-load-time node table. Handles are contiguous and remain valid
        // until the next load_scene call.
        std::vector<donut::engine::SceneGraphNode*> m_nodeHandles;
        std::unordered_map<std::string, uint32_t> m_nodeHandleByName;
        std::unordered_set<std::string> m_ambiguousNodeNames;
        std::vector<MeshSensorLabel> m_meshSensorLabels;
        std::mutex m_sharedTransformMutex;
        std::map<
            std::weak_ptr<rtxns::interop::SharedTransformStream>,
            uint64_t,
            std::owner_less<
                std::weak_ptr<rtxns::interop::SharedTransformStream>>>
            m_sharedTransformEpochs;

        // --- Async batch tracking ---
        struct PendingBatch {
            uint64_t token = 0;
            std::vector<uint32_t> cameraIndices;
            std::vector<uint32_t> ringIndices;
            nvrhi::CommandListHandle commandList;
            nvrhi::EventQueryHandle query;
        };
        std::vector<PendingBatch> m_pendingBatches;
    };

    namespace
    {
        std::mutex g_context_mutex;
        std::shared_ptr<RendererContext> g_context;
    }

    HeadlessPbrScene::HeadlessPbrScene(std::shared_ptr<RendererContext> context)
        : m_impl(std::make_unique<Impl>(std::move(context)))
    {
    }

    HeadlessPbrScene::~HeadlessPbrScene() = default;

    void HeadlessPbrScene::load_scene(const std::filesystem::path& scene_path)
    {
        m_impl->load_scene(scene_path);
    }

    // --- New multi-camera API ---

    uint32_t HeadlessPbrScene::add_camera(
        const std::array<float, 3>& position,
        const std::array<float, 3>& target,
        const std::array<float, 3>& up,
        float fov_degrees,
        uint32_t width,
        uint32_t height,
        float z_near,
        float z_far)
    {
        CameraDesc desc;
        desc.position = position;
        desc.target = target;
        desc.up = up;
        desc.fov_degrees = fov_degrees;
        desc.width = width;
        desc.height = height;
        desc.z_near = z_near;
        desc.z_far = z_far;
        return m_impl->add_camera_slot(desc);
    }

    void HeadlessPbrScene::set_camera_at(
        uint32_t index,
        const std::array<float, 3>& position,
        const std::array<float, 3>& target,
        const std::array<float, 3>& up,
        float fov_degrees,
        uint32_t width,
        uint32_t height,
        float z_near,
        float z_far)
    {
        CameraDesc desc;
        desc.position = position;
        desc.target = target;
        desc.up = up;
        desc.fov_degrees = fov_degrees;
        desc.width = width;
        desc.height = height;
        desc.z_near = z_near;
        desc.z_far = z_far;
        m_impl->set_camera_desc(index, desc);
    }

    uint32_t HeadlessPbrScene::camera_count() const noexcept
    {
        return m_impl->camera_count_impl();
    }

    std::vector<uint8_t> HeadlessPbrScene::render_frame(uint32_t camera_index)
    {
        return m_impl->render_frame_for_index(camera_index);
    }

    std::vector<std::vector<uint8_t>> HeadlessPbrScene::render_frame_batch(const std::vector<uint32_t>& camera_indices)
    {
        // Week 3: sync convenience = submit + wait + read
        if (camera_indices.empty())
            return {};
        uint64_t token = m_impl->submit_frame_batch_impl(camera_indices);
        if (token == 0)
            throw std::runtime_error("Readback ring is busy.");
        return m_impl->read_frame_batch_impl(token);
    }

    uint64_t HeadlessPbrScene::submit_frame_batch(const std::vector<uint32_t>& camera_indices)
    {
        return m_impl->submit_frame_batch_impl(camera_indices);
    }

    bool HeadlessPbrScene::is_batch_ready(uint64_t token) const
    {
        return m_impl->is_batch_ready_impl(token);
    }

    std::vector<std::vector<uint8_t>> HeadlessPbrScene::read_frame_batch(uint64_t token)
    {
        return m_impl->read_frame_batch_impl(token);
    }

    // --- Existing API ---

    void HeadlessPbrScene::set_camera(
        const std::array<float, 3>& position,
        const std::array<float, 3>& target,
        const std::array<float, 3>& up,
        float fov_degrees,
        uint32_t width,
        uint32_t height,
        float z_near,
        float z_far)
    {
        m_impl->set_camera(position, target, up, fov_degrees, width, height, z_near, z_far);
    }

    void HeadlessPbrScene::set_ambient(
        const std::array<float, 3>& top_rgb,
        const std::array<float, 3>& bottom_rgb)
    {
        m_impl->set_ambient(top_rgb, bottom_rgb);
    }

    void HeadlessPbrScene::set_default_light(
        const std::array<float, 3>& direction,
        const std::array<float, 3>& color,
        float irradiance)
    {
        m_impl->set_default_light(direction, color, irradiance);
    }

    void HeadlessPbrScene::update_node_transform(
        const std::string& name,
        const std::vector<float>& matrix_values)
    {
        m_impl->update_node_transform(name, matrix_values);
    }

    uint32_t HeadlessPbrScene::node_handle_count() const noexcept
    {
        return m_impl->node_handle_count();
    }

    std::vector<uint32_t> HeadlessPbrScene::get_node_handles(
        const std::vector<std::string>& names) const
    {
        return m_impl->get_node_handles(names);
    }

    void HeadlessPbrScene::update_node_transforms_batch(
        const std::vector<uint32_t>& handles,
        const std::vector<std::vector<float>>& matrices)
    {
        m_impl->update_node_transforms_batch(handles, matrices);
    }

    HeadlessPbrScene::SharedTransformConsumeToken
        HeadlessPbrScene::consume_shared_transform_slot(
            const std::shared_ptr<rtxns::interop::SharedTransformStream>& stream,
            uint32_t slot,
            const std::vector<uint32_t>& handles)
    {
        return m_impl->consume_shared_transform_slot(stream, slot, handles);
    }

    std::vector<float> HeadlessPbrScene::get_node_world_transform(
        const std::string& name) const
    {
        return m_impl->get_node_world_transform(name);
    }

    std::vector<float> HeadlessPbrScene::get_node_world_transform_by_handle(
        uint32_t handle) const
    {
        return m_impl->get_node_world_transform_by_handle(handle);
    }

    void HeadlessPbrScene::set_node_labels(
        const std::vector<std::string>& node_names,
        const std::vector<uint32_t>& instance_ids,
        const std::vector<uint32_t>& semantic_ids)
    {
        m_impl->set_node_labels(node_names, instance_ids, semantic_ids);
    }

    std::vector<HeadlessPbrScene::SensorFrame> HeadlessPbrScene::render_sensor_batch(
        const std::vector<uint32_t>& camera_indices,
        uint32_t product_mask)
    {
        return m_impl->render_sensor_batch_impl(camera_indices, product_mask);
    }

    HeadlessPbrScene::SceneStats HeadlessPbrScene::get_scene_stats() const
    {
        return m_impl->get_scene_stats();
    }

    void HeadlessPbrScene::set_readback_ring_depth(uint32_t depth)
    {
        m_impl->set_ring_depth(depth);
    }

    uint32_t HeadlessPbrScene::get_readback_ring_depth() const noexcept
    {
        return m_impl->get_ring_depth();
    }

    void HeadlessPbrScene::enable_rt_shadows(bool enable)
    {
        m_impl->enable_rt_shadows(enable);
    }

    void HeadlessPbrScene::enable_shadow_blur(bool enable)
    {
        m_impl->enable_shadow_blur(enable);
    }

    void HeadlessPbrScene::enable_omm(bool enable)
    {
        m_impl->enable_omm(enable);
    }

    void HeadlessPbrScene::set_shadow_samples(uint32_t n)
    {
        m_impl->set_shadow_samples(n);
    }

    void HeadlessPbrScene::enable_omm_stress(bool enable)
    {
        m_impl->enable_omm_stress(enable);
    }

    void HeadlessPbrScene::set_omm_config(uint32_t subdiv, uint32_t format)
    {
        m_impl->set_omm_config(subdiv, format);
    }

    bool HeadlessPbrScene::load_omm_cache(const std::string& path)
    {
        return m_impl->load_omm_cache(path);
    }

    bool HeadlessPbrScene::save_omm_cache(const std::string& path)
    {
        return m_impl->save_omm_cache(path);
    }

    std::vector<uint8_t> HeadlessPbrScene::render_frame()
    {
        return m_impl->render_frame();
    }

    uint32_t HeadlessPbrScene::width() const noexcept
    {
        return m_impl->width();
    }

    uint32_t HeadlessPbrScene::height() const noexcept
    {
        return m_impl->height();
    }

    HeadlessPbrScene::FrameStats HeadlessPbrScene::get_last_frame_stats() const
    {
        return m_impl->lastFrameStats();
    }

    std::shared_ptr<RendererContext> initialize(const ContextInitOptions& options)
    {
        auto context = std::make_shared<RendererContext>(options);
        std::scoped_lock lock(g_context_mutex);
        g_context = context;
        return context;
    }

    void shutdown()
    {
        std::scoped_lock lock(g_context_mutex);
        g_context.reset();
    }

    std::shared_ptr<HeadlessPbrScene> create_scene()
    {
        std::scoped_lock lock(g_context_mutex);
        auto context = g_context;
        if (!context)
        {
            throw std::runtime_error("The RTXNS Donut Python backend is not initialized. Call init(...) first.");
        }

        return std::make_shared<HeadlessPbrScene>(std::move(context));
    }

    std::shared_ptr<rtxns::interop::SharedTransformStream>
        create_shared_transform_stream(uint32_t record_count, uint32_t slot_count)
    {
        std::shared_ptr<RendererContext> context;
        {
            std::scoped_lock lock(g_context_mutex);
            context = g_context;
        }

        if (!context)
        {
            throw std::runtime_error(
                "The RTXNS Donut Python backend is not initialized. Call init(...) first.");
        }
        if (!context->isExternalInteropEnabled())
        {
            throw std::runtime_error(
                "External interop is disabled. Reinitialize with "
                "enable_external_interop=True.");
        }

        nvrhi::IDevice* device = context->device();
        return std::make_shared<rtxns::interop::SharedTransformStream>(
            std::move(context),
            device,
            record_count,
            slot_count);
    }
}
