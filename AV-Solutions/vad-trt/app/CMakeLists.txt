cmake_minimum_required(VERSION 3.14)
set(APP vad)
project(${APP}_app LANGUAGES CXX CUDA)
option(TARGET "x86_64")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "-fPIC -g -O0")
set(CMAKE_CXX_FLAGS "-fPIC -g -O0")

# msgs_wsのインストールパスをCMAKE_PREFIX_PATHに追加
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/msgs_ws/install")

# ROS2の依存関係を追加
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(autoware_perception_msgs REQUIRED)
find_package(autoware_internal_planning_msgs REQUIRED)
find_package(autoware_planning_msgs REQUIRED)
find_package(autoware_tensorrt_common REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(unique_identifier_msgs REQUIRED)
find_package(autoware_utils_uuid REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# Find CUDA package first
find_package(CUDA REQUIRED)

include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

if(${TARGET} MATCHES "aarch64")
    # linux orin configuration remains the same
    set(CMAKE_C_COMPILER   /usr/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
    set(CMAKE_FIND_ROOT_PATH "/usr/aarch64-linux-gnu")
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
    set(CUDA_TOOLKIT_ROOT_DIR     /usr/local/cuda-11.4/)
    set(CUDA_TOOLKIT_TARGET_NAME  "aarch64-linux")
    set(CUDA_TOOLKIT_INCLUDE      /usr/local/cuda-11.4/targets/aarch64-linux/include)
    set(CUDA_CUDART_LIBRARY       /usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so)    
    set(CUDA_HOST_COMPILER  "/usr/bin/aarch64-linux-gnu-g++" CACHE STRING "host compiler" FORCE)
    set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
    set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu)
    set(DEPS ${DEPS} nvcudla cudla cuda cudart)

elseif(${TARGET} MATCHES "x86_64")
    option(TRT_ROOT "$ENV{TRT_ROOT}")
    
    # Ensure CUDA include directories are properly set
    set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
    set(TENSORRT_INCLUDE_DIRS ${TRT_ROOT}/include/)
    set(TENSORRT_LIBRARY_DIRS ${TRT_ROOT}/lib/)
    set(DEPS ${DEPS} cuda cudart)
    
else()
    message(FATAL_ERROR "Unsupported TARGET: ${TARGET}")
endif()

# Common include directories for both targets
include_directories(
    ${CMAKE_SOURCE_DIR}/include 
    ${CMAKE_SOURCE_DIR}/lib 
    ${CMAKE_SOURCE_DIR}/lib/networks
    ${CMAKE_SOURCE_DIR}/lib/cuOSD
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${TENSORRT_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/msgs_ws/install/autoware_internal_planning_msgs/include
)

# Common link directories
link_directories(
    ${CUDA_LIBRARY_DIRS}
    ${TENSORRT_LIBRARY_DIRS}
)

add_definitions(-DENABLE_TEXT_BACKEND_STB)

# CUDA library
add_library(${APP}_cu STATIC
    ${CMAKE_SOURCE_DIR}/lib/visualize.cu
    ${CMAKE_SOURCE_DIR}/lib/cuOSD/cuosd_kernel.cu
)

# CUDA compilation settings
set_target_properties(${APP}_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(${APP}_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86>
)

# CUDAライブラリにもROS2のインクルードディレクトリを追加
target_include_directories(${APP}_cu PUBLIC
    ${rclcpp_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${tf2_ros_INCLUDE_DIRS}
    ${tf2_eigen_INCLUDE_DIRS}
    ${eigen3_cmake_module_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${autoware_perception_msgs_INCLUDE_DIRS}
    ${autoware_internal_planning_msgs_INCLUDE_DIRS}
    ${autoware_planning_msgs_INCLUDE_DIRS}
    ${autoware_tensorrt_common_INCLUDE_DIRS}
    ${geometry_msgs_INCLUDE_DIRS}
    ${rosbag2_cpp_INCLUDE_DIRS}
)

# Main application
add_executable(${APP}_app 
    main.cpp 
    ${CMAKE_SOURCE_DIR}/lib/vad_model.cpp
    ${CMAKE_SOURCE_DIR}/lib/vad_interface.cpp
    ${CMAKE_SOURCE_DIR}/lib/visualize.cpp 
    ${CMAKE_SOURCE_DIR}/lib/networks/tensor.cpp 
    ${CMAKE_SOURCE_DIR}/lib/networks/net.cpp
    ${CMAKE_SOURCE_DIR}/lib/networks/backbone.cpp
    ${CMAKE_SOURCE_DIR}/lib/networks/head.cpp
    ${CMAKE_SOURCE_DIR}/lib/cuOSD/cuosd.cpp
    ${CMAKE_SOURCE_DIR}/lib/cuOSD/textbackend/backend.cpp 
    ${CMAKE_SOURCE_DIR}/lib/cuOSD/textbackend/stb.cpp
)

# Link libraries
target_link_libraries(${APP}_app 
    ${APP}_cu 
    nvinfer 
    ${CUDA_LIBRARIES}
    ${DEPS}
    nlohmann_json::nlohmann_json
    rclcpp::rclcpp
    ${OpenCV_LIBS}
    ${sensor_msgs_LIBRARIES}
    ${tf2_ros_LIBRARIES}
    ${tf2_eigen_LIBRARIES}
    ${eigen3_cmake_module_LIBRARIES}
    ${Eigen3_LIBRARIES}
    ${autoware_perception_msgs_LIBRARIES}
    ${autoware_internal_planning_msgs_LIBRARIES}
    ${autoware_planning_msgs_LIBRARIES}
    ${autoware_tensorrt_common_LIBRARIES}
    ${geometry_msgs_LIBRARIES}
    ${rosbag2_cpp_LIBRARIES}
)

# インクルードディレクトリの追加
target_include_directories(${APP}_app PUBLIC
    ${rclcpp_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${tf2_ros_INCLUDE_DIRS}
    ${tf2_eigen_INCLUDE_DIRS}
    ${eigen3_cmake_module_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${autoware_perception_msgs_INCLUDE_DIRS}
    ${autoware_internal_planning_msgs_INCLUDE_DIRS}
    ${autoware_planning_msgs_INCLUDE_DIRS}
    ${autoware_tensorrt_common_INCLUDE_DIRS}
    ${geometry_msgs_INCLUDE_DIRS}
    ${rosbag2_cpp_INCLUDE_DIRS}
)

# インストール設定
install(TARGETS ${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME}
)

# ament_target_dependenciesでROS2の依存関係を設定
ament_target_dependencies(${APP}_app
    rclcpp
    tf2_ros
    tf2_eigen
    eigen3_cmake_module
    Eigen3
    sensor_msgs
    autoware_perception_msgs
    autoware_internal_planning_msgs
    autoware_planning_msgs
    autoware_tensorrt_common
    geometry_msgs
    rosbag2_cpp
    unique_identifier_msgs
    autoware_utils_uuid
    std_msgs
)
