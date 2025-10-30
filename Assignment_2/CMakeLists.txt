cmake_minimum_required(VERSION 3.20)
project(VisualComputingPipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v0.1.36
)
FetchContent_MakeAvailable(glad)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.7
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_glfw_opengl3 STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui_glfw_opengl3 PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(imgui_glfw_opengl3 PUBLIC glfw)

add_executable(VisualComputingPipeline
    src/main.cpp
    src/Application.cpp
    src/FrameProcessor.cpp
)

target_include_directories(VisualComputingPipeline PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${glad_SOURCE_DIR}/include
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(VisualComputingPipeline PRIVATE
    OpenGL::GL
    glfw
    imgui_glfw_opengl3
    glad
    ${OpenCV_LIBS}
)

target_compile_definitions(VisualComputingPipeline PRIVATE
    IMGUI_IMPL_OPENGL_LOADER_GLAD
)

# Copy shader files next to the executable to simplify deployment.
set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_BINARY_DIR}/shaders)

add_custom_target(copy_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADER_SOURCE_DIR} ${SHADER_BINARY_DIR}
    BYPRODUCTS ${SHADER_BINARY_DIR}
    COMMENT "Copying shaders to build directory"
)

add_dependencies(VisualComputingPipeline copy_shaders)

# Ensure Visual Studio launches the application from the source directory so
# relative paths to shaders and assets keep working.
set_target_properties(VisualComputingPipeline PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
