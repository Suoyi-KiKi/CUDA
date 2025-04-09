cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0104 NEW)

project(MYPROJECT VERSION 1.1 LANGUAGES CXX CUDA)

# 全局设置
set(CMAKE_CUDA_ARCHITECTURES 75)
include_directories(${CMAKE_SOURCE_DIR}/include)

# 依赖配置
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

# 添加主程序
add_subdirectory(src)

# 可选测试构建
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
