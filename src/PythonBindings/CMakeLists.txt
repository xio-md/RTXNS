if (NOT DONUT_WITH_VULKAN)
    message(FATAL_ERROR "The Donut Python backend currently requires DONUT_WITH_VULKAN=ON.")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development Development.Module)

set(RTXNS_LOCAL_PYBIND11_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/third_party/pybind11/include")
if (NOT EXISTS "${RTXNS_LOCAL_PYBIND11_INCLUDE_DIR}/pybind11/pybind11.h")
    message(FATAL_ERROR
        "pybind11 headers were not found at ${RTXNS_LOCAL_PYBIND11_INCLUDE_DIR}.")
endif()

message(STATUS "Using local pybind11 headers from ${RTXNS_LOCAL_PYBIND11_INCLUDE_DIR}")

option(RTXNS_BUILD_LEGACY_RTX_RENDER_PY
    "Build the legacy RtxRenderPy compatibility extension alongside DonutRenderPyNative."
    ON)

set(RTXNS_DONUT_RENDER_RUNTIME_TARGET rtxns_donut_render_runtime)
set(RTXNS_DONUT_RENDER_NATIVE_MODULE_TARGET DonutRenderPyNative)
set(RTXNS_DONUT_RENDER_LEGACY_MODULE_TARGET RtxRenderPy)

add_library(rtxns_pybind11_headers INTERFACE)
target_include_directories(rtxns_pybind11_headers
    INTERFACE
        "${RTXNS_LOCAL_PYBIND11_INCLUDE_DIR}")

add_library(${RTXNS_DONUT_RENDER_RUNTIME_TARGET} STATIC
    headless_pbr.cpp
    headless_pbr.h)

target_include_directories(${RTXNS_DONUT_RENDER_RUNTIME_TARGET}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${RTXNS_DONUT_RENDER_RUNTIME_TARGET}
    PRIVATE
        donut_app
        donut_render
        rtxns_interop
        rtxns_ray_traced_shadow)

set_target_properties(${RTXNS_DONUT_RENDER_RUNTIME_TARGET} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "DonutRender")

function(rtxns_add_python_renderer_module target_name output_name source_file)
    Python_add_library(${target_name} MODULE
        ${source_file})

    target_include_directories(${target_name}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}")

    target_link_libraries(${target_name}
        PRIVATE
            Python::Module
            ${RTXNS_DONUT_RENDER_RUNTIME_TARGET}
            rtxns_pybind11_headers)

    set_target_properties(${target_name} PROPERTIES
        OUTPUT_NAME "${output_name}"
        PREFIX ""
        FOLDER "DonutRender")

    foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(${target_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${config} "${RTXNS_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_${config} "${RTXNS_BINARY_DIR}")
    endforeach()
endfunction()

rtxns_add_python_renderer_module(
    ${RTXNS_DONUT_RENDER_NATIVE_MODULE_TARGET}
    "DonutRenderPyNative"
    py_interface_donut_native.cpp)

if (RTXNS_BUILD_LEGACY_RTX_RENDER_PY)
    rtxns_add_python_renderer_module(
        ${RTXNS_DONUT_RENDER_LEGACY_MODULE_TARGET}
        "RtxRenderPy"
        py_interface.cpp)
endif()
