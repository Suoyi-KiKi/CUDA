# 算法静态库
file(GLOB ALG_SOURCES "algorithm/*.cu")
# add_library(radar_algorithms STATIC ${ALG_SOURCES}) # 设置为静态库
add_library(radar_algorithms SHARED ${ALG_SOURCES})
# 设置动态库属性（可选：版本号、符号导出）
set_target_properties(radar_algorithms PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON      # 确保 CUDA 符号在动态库中解析
    SOVERSION 1                         # 主版本号（兼容性）
    VERSION 1.0.0                       # 完整版本号
)
target_link_libraries(radar_algorithms PUBLIC 
    Eigen3::Eigen
    CUDA::cufft
    CUDA::cublas
    CUDA::cusolver
    CUDA::curand
)

# 主可执行文件
add_executable(main_program main.cu)
target_link_libraries(main_program PRIVATE radar_algorithms)
