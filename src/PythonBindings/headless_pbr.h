#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace rtxns::interop
{
    class SharedTransformStream;
}

namespace rtxns::python
{
    struct ContextInitOptions
    {
        std::filesystem::path runtime_dir;
        std::string backend = "vulkan";
        int device_index = -1;
        bool enable_debug = false;
        bool enable_external_interop = false;
    };

    /// Camera descriptor used by add_camera / set_camera(index).
    struct CameraDesc
    {
        std::array<float, 3> position{};
        std::array<float, 3> target{};
        std::array<float, 3> up{0.0f, 1.0f, 0.0f};
        float fov_degrees = 60.0f;
        uint32_t width = 1024;
        uint32_t height = 768;
        float z_near = 0.1f;
        float z_far = 1000.0f;
    };

    class RendererContext;

    class HeadlessPbrScene
    {
    public:
        explicit HeadlessPbrScene(std::shared_ptr<RendererContext> context);
        ~HeadlessPbrScene();

        void load_scene(const std::filesystem::path& scene_path);

        // --- Legacy single-camera API (backward-compatible, writes camera 0) ---
        void set_camera(
            const std::array<float, 3>& position,
            const std::array<float, 3>& target,
            const std::array<float, 3>& up,
            float fov_degrees,
            uint32_t width,
            uint32_t height,
            float z_near,
            float z_far);

        [[nodiscard]] std::vector<uint8_t> render_frame();

        // --- Multi-camera API (new) ---
        uint32_t add_camera(
            const std::array<float, 3>& position,
            const std::array<float, 3>& target,
            const std::array<float, 3>& up,
            float fov_degrees,
            uint32_t width,
            uint32_t height,
            float z_near = 0.1f,
            float z_far = 1000.0f);

        void set_camera_at(
            uint32_t index,
            const std::array<float, 3>& position,
            const std::array<float, 3>& target,
            const std::array<float, 3>& up,
            float fov_degrees,
            uint32_t width,
            uint32_t height,
            float z_near = 0.1f,
            float z_far = 1000.0f);

        [[nodiscard]] uint32_t camera_count() const noexcept;

        [[nodiscard]] std::vector<uint8_t> render_frame(uint32_t camera_index);

        [[nodiscard]] std::vector<std::vector<uint8_t>> render_frame_batch(const std::vector<uint32_t>& camera_indices);

        // --- Async batch API (Week 3) ---
        [[nodiscard]] uint64_t submit_frame_batch(const std::vector<uint32_t>& camera_indices);
        [[nodiscard]] bool is_batch_ready(uint64_t token) const;
        [[nodiscard]] std::vector<std::vector<uint8_t>> read_frame_batch(uint64_t token);

        void set_ambient(
            const std::array<float, 3>& top_rgb,
            const std::array<float, 3>& bottom_rgb);
        void set_default_light(
            const std::array<float, 3>& direction,
            const std::array<float, 3>& color,
            float irradiance);
        void update_node_transform(
            const std::string& name,
            const std::vector<float>& matrix_values);
        [[nodiscard]] uint32_t node_handle_count() const noexcept;
        [[nodiscard]] std::vector<uint32_t> get_node_handles(
            const std::vector<std::string>& names) const;
        void update_node_transforms_batch(
            const std::vector<uint32_t>& handles,
            const std::vector<std::vector<float>>& matrices);

        struct SharedTransformConsumeToken
        {
            uint64_t epoch = 0;
            uint64_t simulation_step = 0;
            uint64_t timestamp_ns = 0;
            uint32_t slot = 0;
            uint32_t record_count = 0;
        };

        [[nodiscard]] SharedTransformConsumeToken consume_shared_transform_slot(
            const std::shared_ptr<rtxns::interop::SharedTransformStream>& stream,
            uint32_t slot,
            const std::vector<uint32_t>& handles);

        [[nodiscard]] std::vector<float> get_node_world_transform(
            const std::string& name) const;
        [[nodiscard]] std::vector<float> get_node_world_transform_by_handle(
            uint32_t handle) const;

        void set_node_labels(
            const std::vector<std::string>& node_names,
            const std::vector<uint32_t>& instance_ids,
            const std::vector<uint32_t>& semantic_ids);

        enum SensorProduct : uint32_t
        {
            SensorColor = 1u << 0u,
            SensorDepth = 1u << 1u,
            SensorNormal = 1u << 2u,
            SensorInstance = 1u << 3u,
            SensorSemantic = 1u << 4u,
            SensorAll = SensorColor | SensorDepth | SensorNormal |
                SensorInstance | SensorSemantic,
        };

        struct SensorFrame
        {
            uint32_t width = 0;
            uint32_t height = 0;
            std::vector<uint8_t> color_rgba8;
            std::vector<float> depth_linear;
            std::vector<float> normal_world;
            std::vector<uint32_t> instance;
            std::vector<uint32_t> semantic;
        };

        [[nodiscard]] std::vector<SensorFrame> render_sensor_batch(
            const std::vector<uint32_t>& camera_indices,
            uint32_t product_mask = SensorAll);

        struct SceneStats
        {
            uint32_t mesh_instances = 0;
            uint32_t unique_meshes = 0;
            uint32_t unique_geometries = 0;
            uint32_t unique_materials = 0;
            uint64_t unique_vertices = 0;
            uint64_t unique_indices = 0;
            uint32_t shadow_instances = 0;
        };

        [[nodiscard]] SceneStats get_scene_stats() const;

        // --- Ring depth configuration (P0: occupancy protection) ---
        void set_readback_ring_depth(uint32_t depth);
        [[nodiscard]] uint32_t get_readback_ring_depth() const noexcept;

        void enable_rt_shadows(bool enable);
        void enable_shadow_blur(bool enable);
        void enable_omm(bool enable);
        void set_shadow_samples(uint32_t n);
        void enable_omm_stress(bool enable);
        void set_omm_config(uint32_t subdiv, uint32_t format);

        bool load_omm_cache(const std::string& path);
        bool save_omm_cache(const std::string& path);

        [[nodiscard]] uint32_t width() const noexcept;
        [[nodiscard]] uint32_t height() const noexcept;

        struct FrameStats
        {
            double total_ms = 0.0;
            double scene_refresh_cpu_ms = 0.0; // CPU time spent recording Scene::Refresh
            double shadow_as_record_cpu_ms = 0.0; // CPU time spent preparing/recording BLAS/TLAS work
            double sensor_record_cpu_ms = 0.0; // CPU time spent recording requested sensor passes
            double raster_ms = 0.0;       // forward shading (opaque + transparent)
            double blas_build_ms = 0.0;   // BLAS build (first frame only)
            double tlas_build_ms = 0.0;   // TLAS build/update
            double shadow_ray_ms = 0.0;   // ray query compute pass
            double composite_ms = 0.0;    // shadow composite pass
            double readback_ms = 0.0;     // GPU finish + texture readback
            bool   rt_shadows_enabled = false;
            bool   as_built_this_frame = false;
        };

        [[nodiscard]] FrameStats get_last_frame_stats() const;

    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };

    std::shared_ptr<RendererContext> initialize(const ContextInitOptions& options);
    void shutdown();
    std::shared_ptr<HeadlessPbrScene> create_scene();
    std::shared_ptr<rtxns::interop::SharedTransformStream>
        create_shared_transform_stream(uint32_t record_count, uint32_t slot_count = 3);
}
