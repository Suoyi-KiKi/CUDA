#pragma once
// 计时开始和计时结束
// 定义开始和结束宏
#define PROFILE_START(var) auto var##_start = std::chrono::high_resolution_clock::now()
#define PROFILE_END(var, msg)                                                                             \
    auto var##_end = std::chrono::high_resolution_clock::now();                                           \
    auto var##_duration = std::chrono::duration_cast<std::chrono::microseconds>(var##_end - var##_start); \
    std::cout << (msg) << ": " << var##_duration.count() << " μs" << std::endl


//检查CUDA调用是否成功，并在失败时输出错误信息并退出程序。

#define CHECK_CUDA(call)                                                                                                  \
    {                                                                                                                     \
        cudaError_t err = call;                                                                                           \
        if (err != cudaSuccess) {                                                                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                                                           \
        }                                                                                                                 \
    }

//检查cuBLAS库调用是否成功，并在失败时输出错误信息并退出程序
#define CHECK_CUBLAS(func) \
{ \
    cublasStatus_t stat = (func); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS Error at " << __FILE__ << ":" << __LINE__ << " - " << #func << " failed with code: " << stat << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}
//检查cuSOLVER库调用是否成功，并在失败时输出错误信息并退出程序。

#define CHECK_CUSOLVER(func) \
{ \
    cusolverStatus_t stat = (func); \
    if (stat != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "CUSOLVER Error at " << __FILE__ << ":" << __LINE__ << " - " << #func << " failed with code: " << stat << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}
