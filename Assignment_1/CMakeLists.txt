cmake_minimum_required(VERSION 3.15)
project(panorama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

file(GLOB SRC_FILES src/*.cpp)
add_executable(panorama ${SRC_FILES})

target_link_libraries(panorama ${OpenCV_LIBS})
