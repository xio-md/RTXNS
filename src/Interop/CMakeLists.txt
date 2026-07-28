add_library(rtxns_interop STATIC
    SharedTransformStream.cpp
    SharedTransformStream.h)

target_include_directories(rtxns_interop
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(rtxns_interop
    PUBLIC
        nvrhi)

if (TARGET nvrhi_vk)
    target_link_libraries(rtxns_interop PRIVATE nvrhi_vk)
endif()

if (TARGET Vulkan-Headers)
    target_link_libraries(rtxns_interop PRIVATE Vulkan-Headers)
elseif (TARGET Vulkan::Headers)
    target_link_libraries(rtxns_interop PRIVATE Vulkan::Headers)
endif()

set_target_properties(rtxns_interop PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "DonutRender")
