# 独立测试可执行
add_executable(mtd_test test_mtd.cu)

# 链接主算法库和依赖
target_link_libraries(mtd_test PRIVATE
    radar_algorithms  # 来自src的库
    CUDA::cufft
)
# 设置测试的工作目录为二进制目录
add_test(
    NAME mtd_test
    COMMAND $<TARGET_FILE:mtd_test>  # 使用生成器表达式确保路径正确
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}  # 改为构建目录
)
