# 独立测试可执行
add_executable(cfar_test test_cfar.cu)

# 链接主算法库和依赖
target_link_libraries(cfar_test PRIVATE
    radar_algorithms  # 来自src的库
    CUDA::cufft
)

# 通过CMakeLists.txt的设定直接避免内部自定义查找运行路径的代码
target_compile_definitions(cfar_test PRIVATE SOURCE_DIR="${CMAKE_SOURCE_DIR}")
# 设置测试的工作目录为二进制目录
add_test(
    NAME cfar_test
    COMMAND $<TARGET_FILE:cfar_test>  # 使用生成器表达式确保路径正确
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}  # 改为构建目录
)
