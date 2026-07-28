#pragma once

#include "headless_pbr.h"
#include "../Interop/SharedTransformStream.h"

#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_set>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

inline void bind_rtxns_headless_pbr_module(py::module_ &m)
{
    m.doc() = "Headless Vulkan PBR renderer bindings for the RTXNS Donut backend.";

    using rtxns::interop::OwnedWin32Handle;
    using rtxns::interop::SharedTransformStream;

    py::class_<OwnedWin32Handle, std::shared_ptr<OwnedWin32Handle>>(
        m,
        "OwnedWin32Handle",
        "Owns one duplicated Win32 handle until close(), detach(), or destruction.")
        .def_property_readonly("value", &OwnedWin32Handle::value)
        .def_property_readonly("closed", &OwnedWin32Handle::closed)
        .def("detach",
            &OwnedWin32Handle::detach,
            "Transfer ownership to the caller and return the raw handle value.")
        .def("close", &OwnedWin32Handle::close)
        .def("__int__", &OwnedWin32Handle::value)
        .def("__enter__",
            [](const std::shared_ptr<OwnedWin32Handle>& self)
            {
                return self;
            })
        .def("__exit__",
            [](OwnedWin32Handle& self, py::object, py::object, py::object)
            {
                self.close();
            });

    py::class_<SharedTransformStream, std::shared_ptr<SharedTransformStream>>(
        m,
        "SharedTransformStream",
        "Triple-buffered Vulkan/CUDA transform transport with binary semaphore handoff.")
        .def_property_readonly("record_count", &SharedTransformStream::record_count)
        .def_property_readonly("slot_count", &SharedTransformStream::slot_count)
        .def_property_readonly(
            "slot_payload_size",
            &SharedTransformStream::slot_payload_size)
        .def_property_readonly("slot_stride", &SharedTransformStream::slot_stride)
        .def_property_readonly("logical_size", &SharedTransformStream::logical_size)
        .def_property_readonly(
            "allocation_size",
            &SharedTransformStream::allocation_size)
        .def_property_readonly("closed", &SharedTransformStream::closed)
        .def_property_readonly("poisoned", &SharedTransformStream::poisoned)
        .def("descriptor",
            [](const SharedTransformStream& self)
            {
                py::dict descriptor;
                descriptor["abi"] = "flora.shared_transform.v1";
                descriptor["abi_major"] = SharedTransformStream::kAbiMajor;
                descriptor["abi_minor"] = SharedTransformStream::kAbiMinor;
                descriptor["header_magic"] = SharedTransformStream::kHeaderMagic;
                descriptor["header_size"] = SharedTransformStream::kHeaderSize;
                descriptor["record_count"] = self.record_count();
                descriptor["record_stride"] = SharedTransformStream::kRecordStride;
                descriptor["record_layout"] = "row_major_float3x4";
                descriptor["slot_count"] = self.slot_count();
                descriptor["slot_payload_size"] = self.slot_payload_size();
                descriptor["slot_stride"] = self.slot_stride();
                descriptor["slot_alignment"] =
                    SharedTransformStream::kSlotAlignment;
                descriptor["logical_size"] = self.logical_size();
                descriptor["allocation_size"] = self.allocation_size();
                descriptor["memory_offset"] = self.memory_offset();
                descriptor["dedicated_allocation"] = true;
                descriptor["memory_handle_type"] = "opaque_win32";
                descriptor["semaphore_handle_type"] = "opaque_win32";
                descriptor["semaphore_type"] = "binary";
                descriptor["memory_handle"] = self.duplicate_memory_handle();

                const auto uuid = self.device_uuid();
                descriptor["device_uuid"] = py::bytes(
                    reinterpret_cast<const char*>(uuid.data()),
                    static_cast<py::ssize_t>(uuid.size()));
                std::ostringstream uuid_hex;
                uuid_hex << std::hex << std::setfill('0');
                for (uint8_t byte : uuid)
                    uuid_hex << std::setw(2) << static_cast<unsigned int>(byte);
                descriptor["device_uuid_hex"] = uuid_hex.str();

                py::dict header_layout;
                header_layout["magic"] = py::make_tuple(0, "uint32");
                header_layout["abi_major"] = py::make_tuple(4, "uint16");
                header_layout["abi_minor"] = py::make_tuple(6, "uint16");
                header_layout["header_size"] = py::make_tuple(8, "uint32");
                header_layout["record_count"] = py::make_tuple(12, "uint32");
                header_layout["epoch"] = py::make_tuple(16, "uint64");
                header_layout["simulation_step"] = py::make_tuple(24, "uint64");
                header_layout["timestamp_ns"] = py::make_tuple(32, "uint64");
                header_layout["flags"] = py::make_tuple(40, "uint32");
                descriptor["header_layout"] = std::move(header_layout);

                py::list slots;
                for (uint32_t index = 0; index < self.slot_count(); ++index)
                {
                    py::dict slot;
                    slot["index"] = index;
                    slot["offset"] = self.slot_offset(index);
                    slot["ready_handle"] = self.duplicate_ready_handle(index);
                    slot["consumed_handle"] =
                        self.duplicate_consumed_handle(index);
                    slots.append(std::move(slot));
                }
                descriptor["slots"] = std::move(slots);
                return descriptor;
            },
            "Return the ABI descriptor with fresh caller-owned Win32 handle duplicates.")
        .def("debug_consume_slot",
            [](SharedTransformStream& self, uint32_t slot)
            {
                auto bytes = [&]()
                {
                    py::gil_scoped_release release;
                    return self.debug_consume_slot(slot);
                }();
                return py::bytes(
                    reinterpret_cast<const char*>(bytes.data()),
                    static_cast<py::ssize_t>(bytes.size()));
            },
            py::arg("slot"))
        .def("debug_consume_slot_u32",
            [](SharedTransformStream& self, uint32_t slot)
            {
                py::gil_scoped_release release;
                return self.debug_consume_slot_u32(slot);
            },
            py::arg("slot"))
        .def("debug_publish_slot",
            [](SharedTransformStream& self, uint32_t slot, py::bytes payload)
            {
                const std::string bytes = payload;
                const std::vector<uint8_t> data(bytes.begin(), bytes.end());
                py::gil_scoped_release release;
                self.debug_publish_slot(slot, data);
            },
            py::arg("slot"),
            py::arg("payload"))
        .def("close",
            [](SharedTransformStream& self)
            {
                py::gil_scoped_release release;
                self.close();
            })
        .def("__enter__",
            [](const std::shared_ptr<SharedTransformStream>& self)
            {
                return self;
            })
        .def("__exit__",
            [](SharedTransformStream& self, py::object, py::object, py::object)
            {
                py::gil_scoped_release release;
                self.close();
            });

    py::class_<rtxns::python::HeadlessPbrScene, std::shared_ptr<rtxns::python::HeadlessPbrScene>>(m, "Scene")
        .def("load_scene",
            [](rtxns::python::HeadlessPbrScene &self, const std::string &path)
            {
                py::gil_scoped_release release;
                self.load_scene(path);
            },
            py::arg("path"))
        .def("set_camera",
            &rtxns::python::HeadlessPbrScene::set_camera,
            py::arg("position"),
            py::arg("target"),
            py::arg("up"),
            py::arg("fov_degrees"),
            py::arg("width"),
            py::arg("height"),
            py::arg("z_near") = 0.1f,
            py::arg("z_far") = 1000.0f)
        .def("set_ambient",
            &rtxns::python::HeadlessPbrScene::set_ambient,
            py::arg("top_rgb"),
            py::arg("bottom_rgb"))
        .def("set_default_light",
            &rtxns::python::HeadlessPbrScene::set_default_light,
            py::arg("direction"),
            py::arg("color") = std::array<float, 3>{1.0f, 1.0f, 1.0f},
            py::arg("irradiance") = 2.0f)
        .def("update_node_transform",
            &rtxns::python::HeadlessPbrScene::update_node_transform,
            py::arg("name"),
            py::arg("matrix_values"))
        .def_property_readonly("node_handle_count",
            &rtxns::python::HeadlessPbrScene::node_handle_count)
        .def("get_node_handles",
            &rtxns::python::HeadlessPbrScene::get_node_handles,
            py::arg("names"))
        .def("update_node_transforms_batch",
            &rtxns::python::HeadlessPbrScene::update_node_transforms_batch,
            py::arg("handles"),
            py::arg("matrices"))
        .def("consume_shared_transform_slot",
            [](rtxns::python::HeadlessPbrScene& self,
                const std::shared_ptr<SharedTransformStream>& stream,
                uint32_t slot,
                const std::vector<uint32_t>& handles)
            {
                const auto token = [&]()
                {
                    py::gil_scoped_release release;
                    return self.consume_shared_transform_slot(
                        stream,
                        slot,
                        handles);
                }();

                py::dict result;
                result["epoch"] = token.epoch;
                result["simulation_step"] = token.simulation_step;
                result["timestamp_ns"] = token.timestamp_ns;
                result["slot"] = token.slot;
                result["record_count"] = token.record_count;
                return result;
            },
            py::arg("stream"),
            py::arg("slot"),
            py::arg("handles"),
            "Consume one FST1 slot entirely in C++ and update mapped scene nodes.")
        .def("get_node_world_transform",
            &rtxns::python::HeadlessPbrScene::get_node_world_transform,
            py::arg("name"))
        .def("get_node_world_transform_by_handle",
            &rtxns::python::HeadlessPbrScene::get_node_world_transform_by_handle,
            py::arg("handle"))
        .def("set_node_labels",
            &rtxns::python::HeadlessPbrScene::set_node_labels,
            py::arg("node_names"),
            py::arg("instance_ids"),
            py::arg("semantic_ids"))
        .def("get_scene_stats",
            [](const rtxns::python::HeadlessPbrScene &self)
            {
                const auto stats = self.get_scene_stats();
                py::dict result;
                result["mesh_instances"] = stats.mesh_instances;
                result["unique_meshes"] = stats.unique_meshes;
                result["unique_geometries"] = stats.unique_geometries;
                result["unique_materials"] = stats.unique_materials;
                result["unique_vertices"] = stats.unique_vertices;
                result["unique_indices"] = stats.unique_indices;
                result["shadow_instances"] = stats.shadow_instances;
                return result;
            })
        .def("enable_rt_shadows",
            &rtxns::python::HeadlessPbrScene::enable_rt_shadows,
            py::arg("enable"))
        .def("enable_shadow_blur",
            &rtxns::python::HeadlessPbrScene::enable_shadow_blur,
            py::arg("enable"))
        .def("enable_omm",
            &rtxns::python::HeadlessPbrScene::enable_omm,
            py::arg("enable"))
        .def("set_shadow_samples",
            &rtxns::python::HeadlessPbrScene::set_shadow_samples,
            py::arg("n"))
        .def("enable_omm_stress",
            &rtxns::python::HeadlessPbrScene::enable_omm_stress,
            py::arg("enable"))
        .def("set_omm_config",
            &rtxns::python::HeadlessPbrScene::set_omm_config,
            py::arg("subdiv"), py::arg("format"))
        .def("load_omm_cache",
            [](rtxns::python::HeadlessPbrScene &self, const std::string& path)
            {
                py::gil_scoped_release release;
                return self.load_omm_cache(path);
            },
            py::arg("path"))
        .def("save_omm_cache",
            [](rtxns::python::HeadlessPbrScene &self, const std::string& path)
            {
                py::gil_scoped_release release;
                return self.save_omm_cache(path);
            },
            py::arg("path"))
        .def("add_camera",
            &rtxns::python::HeadlessPbrScene::add_camera,
            py::arg("position"),
            py::arg("target"),
            py::arg("up"),
            py::arg("fov_degrees"),
            py::arg("width"),
            py::arg("height"),
            py::arg("z_near") = 0.1f,
            py::arg("z_far") = 1000.0f)
        .def("set_camera_at",
            &rtxns::python::HeadlessPbrScene::set_camera_at,
            py::arg("index"),
            py::arg("position"),
            py::arg("target"),
            py::arg("up"),
            py::arg("fov_degrees"),
            py::arg("width"),
            py::arg("height"),
            py::arg("z_near") = 0.1f,
            py::arg("z_far") = 1000.0f)
        .def_property_readonly("camera_count", &rtxns::python::HeadlessPbrScene::camera_count)
        .def("render_frame",
            [](rtxns::python::HeadlessPbrScene &self, int camera_index)
            {
                auto pixels = [&]()
                {
                    py::gil_scoped_release release;
                    return self.render_frame(static_cast<uint32_t>(camera_index));
                }();

                return py::bytes(
                    reinterpret_cast<const char *>(pixels.data()),
                    static_cast<py::ssize_t>(pixels.size()));
            },
            py::arg("camera_index") = 0)
        .def("render_frame_batch",
            [](rtxns::python::HeadlessPbrScene &self, const std::vector<uint32_t>& indices)
            {
                auto frames = [&]()
                {
                    py::gil_scoped_release release;
                    return self.render_frame_batch(indices);
                }();

                py::list out;
                for (const auto& pixels : frames)
                    out.append(py::bytes(
                        reinterpret_cast<const char*>(pixels.data()),
                        static_cast<py::ssize_t>(pixels.size())));
                return out;
            },
            py::arg("camera_indices"))
        .def("render_sensor_batch",
            [](rtxns::python::HeadlessPbrScene &self,
                const std::vector<uint32_t>& indices,
                const std::vector<std::string>& products)
            {
                uint32_t mask = 0;
                std::unordered_set<std::string> uniqueProducts;
                for (const auto& product : products)
                {
                    if (!uniqueProducts.insert(product).second)
                        continue;
                    if (product == "color")
                        mask |= rtxns::python::HeadlessPbrScene::SensorColor;
                    else if (product == "depth")
                        mask |= rtxns::python::HeadlessPbrScene::SensorDepth;
                    else if (product == "normal")
                        mask |= rtxns::python::HeadlessPbrScene::SensorNormal;
                    else if (product == "instance")
                        mask |= rtxns::python::HeadlessPbrScene::SensorInstance;
                    else if (product == "semantic")
                        mask |= rtxns::python::HeadlessPbrScene::SensorSemantic;
                    else
                        throw py::value_error("Unknown sensor product: " + product);
                }

                auto frames = [&]()
                {
                    py::gil_scoped_release release;
                    return self.render_sensor_batch(indices, mask);
                }();

                py::list output;
                for (const auto& frame : frames)
                {
                    py::dict item;
                    item["width"] = frame.width;
                    item["height"] = frame.height;
                    if (!frame.color_rgba8.empty())
                        item["color"] = py::bytes(
                            reinterpret_cast<const char*>(frame.color_rgba8.data()),
                            static_cast<py::ssize_t>(frame.color_rgba8.size()));
                    if (!frame.depth_linear.empty())
                        item["depth"] = py::bytes(
                            reinterpret_cast<const char*>(frame.depth_linear.data()),
                            static_cast<py::ssize_t>(
                                frame.depth_linear.size() * sizeof(float)));
                    if (!frame.normal_world.empty())
                        item["normal"] = py::bytes(
                            reinterpret_cast<const char*>(frame.normal_world.data()),
                            static_cast<py::ssize_t>(
                                frame.normal_world.size() * sizeof(float)));
                    if (!frame.instance.empty())
                        item["instance"] = py::bytes(
                            reinterpret_cast<const char*>(frame.instance.data()),
                            static_cast<py::ssize_t>(
                                frame.instance.size() * sizeof(uint32_t)));
                    if (!frame.semantic.empty())
                        item["semantic"] = py::bytes(
                            reinterpret_cast<const char*>(frame.semantic.data()),
                            static_cast<py::ssize_t>(
                                frame.semantic.size() * sizeof(uint32_t)));
                    output.append(std::move(item));
                }
                return output;
            },
            py::arg("camera_indices"),
            py::arg("products") = std::vector<std::string>{
                "color", "depth", "normal", "instance", "semantic"})
        .def("submit_frame_batch",
            &rtxns::python::HeadlessPbrScene::submit_frame_batch,
            py::arg("camera_indices"))
        .def("is_batch_ready",
            &rtxns::python::HeadlessPbrScene::is_batch_ready,
            py::arg("token"))
        .def("read_frame_batch",
            [](rtxns::python::HeadlessPbrScene &self, uint64_t token)
            {
                auto frames = [&]()
                {
                    py::gil_scoped_release release;
                    return self.read_frame_batch(token);
                }();

                py::list out;
                for (const auto& pixels : frames)
                    out.append(py::bytes(
                        reinterpret_cast<const char*>(pixels.data()),
                        static_cast<py::ssize_t>(pixels.size())));
                return out;
            },
            py::arg("token"))
        .def("set_readback_ring_depth",
            &rtxns::python::HeadlessPbrScene::set_readback_ring_depth,
            py::arg("depth"))
        .def_property_readonly("readback_ring_depth",
            &rtxns::python::HeadlessPbrScene::get_readback_ring_depth)
        .def("get_last_frame_stats",
            [](const rtxns::python::HeadlessPbrScene &self)
            {
                const auto& s = self.get_last_frame_stats();
                py::dict d;
                d["total_ms"] = s.total_ms;
                d["scene_refresh_cpu_ms"] = s.scene_refresh_cpu_ms;
                d["shadow_as_record_cpu_ms"] = s.shadow_as_record_cpu_ms;
                d["sensor_record_cpu_ms"] = s.sensor_record_cpu_ms;
                d["raster_ms"] = s.raster_ms;
                d["blas_build_ms"] = s.blas_build_ms;
                d["tlas_build_ms"] = s.tlas_build_ms;
                d["shadow_ray_ms"] = s.shadow_ray_ms;
                d["composite_ms"] = s.composite_ms;
                d["readback_ms"] = s.readback_ms;
                d["rt_shadows_enabled"] = s.rt_shadows_enabled;
                d["as_built_this_frame"] = s.as_built_this_frame;
                return d;
            })
        .def_property_readonly("width", &rtxns::python::HeadlessPbrScene::width)
        .def_property_readonly("height", &rtxns::python::HeadlessPbrScene::height);

    m.def("init",
        [](const std::string &runtime_dir,
            const std::string &backend,
            int device_index,
            bool enable_debug,
            bool enable_external_interop)
        {
            rtxns::python::ContextInitOptions options;
            options.runtime_dir = runtime_dir;
            options.backend = backend;
            options.device_index = device_index;
            options.enable_debug = enable_debug;
            options.enable_external_interop = enable_external_interop;

            py::gil_scoped_release release;
            rtxns::python::initialize(options);
        },
        py::arg("runtime_dir") = "",
        py::arg("backend") = "vulkan",
        py::arg("device_index") = -1,
        py::arg("enable_debug") = false,
        py::arg("enable_external_interop") = false);

    m.def("create_scene", &rtxns::python::create_scene);
    m.def("create_shared_transform_stream",
        [](uint32_t record_count, uint32_t slot_count)
        {
            py::gil_scoped_release release;
            return rtxns::python::create_shared_transform_stream(
                record_count,
                slot_count);
        },
        py::arg("record_count"),
        py::arg("slot_count") = 3);

    m.def("destroy",
        []()
        {
            py::gil_scoped_release release;
            rtxns::python::shutdown();
        });
}
