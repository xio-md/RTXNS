include(${CMAKE_SOURCE_DIR}/external/donut/compileshaders.cmake)

set(LIB_NAME rtxns_ray_traced_shadow)
set(RTXNS_OMM_DIR "${CMAKE_SOURCE_DIR}/../OMM"
    CACHE PATH "Path to the NVIDIA Opacity Micro-Map SDK checkout")

# ---- Shader compilation ----

set(shader_includes
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
)

file(GLOB rtxns_shadow_shaders_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.hlsl"
)

if(DONUT_WITH_STATIC_SHADERS)
    set(shader_output_format "HEADER")
else()
    set(shader_output_format "BINARY")
endif()

donut_compile_shaders_all_platforms(
    TARGET ${LIB_NAME}_shaders
    PROJECT_NAME "RTXNS"
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shaders.cfg
    INCLUDES ${shader_includes}
    FOLDER "RayTracedShadow"
    OUTPUT_BASE ${CMAKE_CURRENT_BINARY_DIR}/shaders
    OUTPUT_FORMAT ${shader_output_format}
    SOURCES ${rtxns_shadow_shaders_sources}
)

# ---- Static library ----

add_library(${LIB_NAME} STATIC
    ShadowTypes.h
    SceneGeometryProvider.h
    SceneGeometryProvider.cpp
    AccelerationStructure.h
    AccelerationStructure.cpp
    RayTracedShadowPass.h
    RayTracedShadowPass.cpp
    OMMBaker.h
    OMMBaker.cpp
)

target_include_directories(${LIB_NAME}
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(${LIB_NAME}
    PUBLIC
        donut_render
        nvrhi
    PRIVATE
        "${RTXNS_OMM_DIR}/build/libraries/omm-lib/$<CONFIG>/omm-bake.lib")

target_include_directories(${LIB_NAME}
    PRIVATE
        "${RTXNS_OMM_DIR}/libraries/omm-lib/include")

target_compile_definitions(${LIB_NAME}
    PRIVATE
        VK_KHR_OPACITY_MICROMAP=1)

add_dependencies(${LIB_NAME} ${LIB_NAME}_shaders)

set_target_properties(${LIB_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "RayTracedShadow")
