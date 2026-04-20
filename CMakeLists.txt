cmake_minimum_required(VERSION 3.20)
project(vkrt CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin /usr/bin /usr/local/bin)
if(GLSLC)
    set(GLSL_COMPILER ${GLSLC})
    set(GLSL_FLAGS    --target-env=vulkan1.2)
    set(GLSL_OUT_FLAG -o)
else()
    find_program(GLSLANG glslangValidator HINTS $ENV{VULKAN_SDK}/bin /usr/bin)
    if(NOT GLSLANG)
        message(FATAL_ERROR "Neither glslc nor glslangValidator found. Install the Vulkan SDK.")
    endif()
    set(GLSL_COMPILER ${GLSLANG})
    set(GLSL_FLAGS    --target-env vulkan1.2 -V)
    set(GLSL_OUT_FLAG -o)
    message(STATUS "Using glslangValidator as shader compiler")
endif()

set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_BIN_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BIN_DIR})

set(SHADER_SOURCES
    ${SHADER_SRC_DIR}/raygen.rgen
    ${SHADER_SRC_DIR}/miss.rmiss
    ${SHADER_SRC_DIR}/closesthit.rchit
)

set(SHADER_SPIRV "")
foreach(SRC ${SHADER_SOURCES})
    get_filename_component(NAME ${SRC} NAME)
    set(SPV ${SHADER_BIN_DIR}/${NAME}.spv)
    add_custom_command(
        OUTPUT  ${SPV}
        COMMAND ${GLSL_COMPILER} ${GLSL_FLAGS} ${SRC} ${GLSL_OUT_FLAG} ${SPV}
        DEPENDS ${SRC}
        COMMENT "Compiling ${NAME}"
        VERBATIM
    )
    list(APPEND SHADER_SPIRV ${SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SHADER_SPIRV})

add_executable(vkrt
    src/main.cpp
    src/Image.cpp
    src/Renderer.cpp
)

add_dependencies(vkrt shaders)

target_include_directories(vkrt PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    src
)

# SHADER_DIR baked in at compile time so shaders are always found
target_compile_definitions(vkrt PRIVATE
    SHADER_DIR="${SHADER_BIN_DIR}"
)

target_link_libraries(vkrt PRIVATE
    ${Vulkan_LIBRARIES}
)

target_compile_options(vkrt PRIVATE -Wall -Wextra -O2)
