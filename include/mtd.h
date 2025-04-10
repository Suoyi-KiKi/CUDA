#ifndef _MTD_H_
#define _MTD_H_

#include "common_headers.h"

constexpr int BLOCK_SIZE = 256;

class FFTShift2DHandler {
    public:
        FFTShift2DHandler(int Q, int SampleNumber);
        ~FFTShift2DHandler();
        cuFloatComplex* execute(cuFloatComplex* d_data_in, cudaStream_t stream = 0);
    
    private:
        int Q, SampleNumber;
        cuFloatComplex* d_data;
        cuFloatComplex* d_abs_result;
        // Disable copying and assignment
        FFTShift2DHandler(const FFTShift2DHandler&) = delete;
        FFTShift2DHandler& operator=(const FFTShift2DHandler&) = delete;
    };
    
    // ---------------------- FFTHandler 定义 ----------------------
    class FFTHandler {
    public:
        FFTHandler(int Q, int SampleNumber);
        ~FFTHandler();
        cuFloatComplex* execute(__half* d_data_in, cudaStream_t stream = 0);
        cuFloatComplex* execute(cuFloatComplex* d_data_in, cudaStream_t stream = 0);
    
    private:
        int Q, SampleNumber;
        cuFloatComplex* d_data;
        cuFloatComplex *d_data_in_float, *d_data_out_float;  // 单精度计算缓冲区
        cufftHandle plan;
        // Disable copying and assignment
        FFTHandler(const FFTHandler&) = delete;
        FFTHandler& operator=(const FFTHandler&) = delete;
    };


void MTD_CUDA_SIM_2D_C_Style(__half* d_MFout, cuFloatComplex* h_MTDabsout, int rows, int cols) ;
void MTD_CUDA_SIM_2D_C_Style(cuFloatComplex* d_SVDout, cuFloatComplex* h_MTDabsout, int rows, int cols) ;


#endif