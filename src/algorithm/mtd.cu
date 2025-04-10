#include "mtd.h"
#include "tools.h"
#include "dataProcess.h"

// ---------------------- 修改FFTShift Kernel ----------------------
__global__ void fftShiftKernel2d(cuFloatComplex* data,  int q, int k) {
    const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 每行一个block
    const int halfQ = q >> 1;  // 用位移替代除法

 

    if (row_idx < k) {
        // 只交换半行数据，避免重复操作
        for (int swap_pos = 0; swap_pos < halfQ; ++swap_pos) {
            const int front_idx = row_idx * q + swap_pos;
            const int back_idx = front_idx + halfQ;

            // 执行交换操作
            cuFloatComplex tmp = data[front_idx];
            data[front_idx] = data[back_idx];
            data[back_idx] = tmp;

           

            // // 计算模值（使用快速近似计算）
            // abs_result[front_idx] = sqrtf(data[front_idx].x * data[front_idx].x + data[front_idx].y * data[front_idx].y);
            // abs_result[back_idx] = sqrtf(data[back_idx].x * data[back_idx].x + data[back_idx].y * data[back_idx].y);
        }
    }
}

// ---------------------- FFTHandler 修改 ----------------------
FFTHandler::FFTHandler(int Q, int SampleNumber) : Q(Q), SampleNumber(SampleNumber) {
    // 分配设备内存
    // 嵌入式设备：cudaHostAlloc(页锁定内存)
    cudaMalloc(&d_data, sizeof(cuFloatComplex) * Q * SampleNumber * 2);
    d_data_in_float = d_data;
    d_data_out_float = d_data_in_float + Q * SampleNumber;

    // 创建FFT计划
    cufftPlan1d(&plan, Q, CUFFT_C2C, SampleNumber);
}

FFTHandler::~FFTHandler() {
    // 释放设备内存
    cudaFree(d_data);

    // 释放FFT计划
    cufftDestroy(plan);
}
// 执行函数修改（核心修改部分）
cuFloatComplex* FFTHandler::execute(__half* d_data_in, cudaStream_t stream) {
    // 转换半精度到单精度（使用CUDA内核）
    dim3 block(BLOCK_SIZE);
    dim3 grid((SampleNumber + BLOCK_SIZE - 1) / BLOCK_SIZE);
    convertHalfToFloat_Optimized<<<grid, block, 0, stream>>>(d_data_in, d_data_in_float, Q * SampleNumber);
    // convertHalfToFloat<<<grid, block, 0, stream>>>(d_data_in, d_data_in_float, Q * SampleNumber);

    cufftSetStream(plan, stream);
    // 执行FFT
    cufftExecC2C(plan, d_data_in_float, d_data_out_float, CUFFT_FORWARD);

    return d_data_out_float;
}

cuFloatComplex* FFTHandler::execute(cuFloatComplex* d_data_in_float, cudaStream_t stream) {
    cufftSetStream(plan, stream);
    cufftExecC2C(plan, d_data_in_float, d_data_out_float, CUFFT_FORWARD);  // 直接计算
    return d_data_out_float;
}


// ---------------------- FFTShift2DHandler 修改 ----------------------
FFTShift2DHandler::FFTShift2DHandler(int Q, int SampleNumber) : Q(Q), SampleNumber(SampleNumber) {
    // 分配设备内存（使用默认设备）
    cudaMalloc(&d_data, sizeof(cuFloatComplex) * Q * SampleNumber);
    cudaMalloc(&d_abs_result, sizeof(cuFloatComplex) * Q * SampleNumber);
}

FFTShift2DHandler::~FFTShift2DHandler() {
    // 释放设备内存
    cudaFree(d_data);
    cudaFree(d_abs_result);
}

cuFloatComplex* FFTShift2DHandler::execute(cuFloatComplex* d_data_in, cudaStream_t stream) {
    // 如果输入数据与内部缓冲区不同，拷贝数据
    if (d_data_in != d_data) {
        cudaMemcpyAsync(d_data, d_data_in, sizeof(cuFloatComplex) * Q * SampleNumber, cudaMemcpyDeviceToDevice, stream);
    }

    // Set thread and block configuration for CUDA kernel
    dim3 block(256);
    dim3 grid((SampleNumber + 256 - 1) / 256);

    // Call the CUDA kernel to perform FFT shift
    fftShiftKernel2d<<<grid, block, 0, stream>>>(d_data,  Q, SampleNumber);

    return d_data;
}


void MTD_CUDA_SIM_2D_C_Style(__half* d_MFout,  cuFloatComplex* h_MTDabsout, int rows, int cols) {
    // 初始化处理器
    FFTHandler fftHandler(rows, cols);
    FFTShift2DHandler fftShift2Dhandler(rows, cols);

    // 执行流水线（全部同步到同一个流）

    cuFloatComplex* fft_result = fftHandler.execute(d_MFout);
    cuFloatComplex* d_abs_res = fftShift2Dhandler.execute(fft_result);

    // 同步并拷贝结果
    cudaError_t err = cudaMemcpyAsync(h_MTDabsout, d_abs_res, sizeof(cuFloatComplex) * rows * cols, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "寄了" << std::endl;
    }

}

void MTD_CUDA_SIM_2D_C_Style(cuFloatComplex* d_SVDout,  cuFloatComplex* h_MTDabsout, int rows, int cols) {
    // 初始化处理器
    FFTHandler fftHandler(rows, cols);

    FFTShift2DHandler fftShift2Dhandler(rows, cols);



    // 执行流水线（全部同步到同一个流）
    cuFloatComplex* fft_result = fftHandler.execute(d_SVDout);
    cuFloatComplex* d_abs_res = fftShift2Dhandler.execute(fft_result);



    // 同步并拷贝结果
    cudaError_t err = cudaMemcpyAsync(h_MTDabsout, d_abs_res, sizeof(cuFloatComplex) * rows * cols, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "寄了" << std::endl;
    }
}


