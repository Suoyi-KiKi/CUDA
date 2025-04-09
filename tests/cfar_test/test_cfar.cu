#include "common_headers.h"
#include "dataProcess.h"
#include "cfar.h"
#include "tools.h"


int main() {

    const int rows = 1100;
    const int cols = 182;
    const int num_elements = rows * cols;

    const std::string InputPath = "/home/kiki/MYCUDA/Bin/MTDOUT.bin";
    const std::string outputPath = "/home/kiki/MYCUDA/Bin/CFAROUT.bin";
    
    // 配置 CFAR 参数（可自行调整）
    int guard_rows = 15, guard_cols = 5;
    int train_rows = 15, train_cols = 10;
    // float threshold_scale = 1.4f;
    float pfa = pow(10.0f, -6.5f);
    float threshold_scale = (pow(pfa, -1.0f / 10920.0f) - 1.0f) * 10920.0f;
    
    // 分配主机和设备内存
    cuFloatComplex* h_in = (cuFloatComplex*) malloc(num_elements * sizeof(cuFloatComplex));
    cuFloatComplex* h_out = (cuFloatComplex*) malloc(num_elements * sizeof(cuFloatComplex));
    cuFloatComplex* d_in;
    cuFloatComplex* d_out;
    cudaMalloc((void**)&d_in, num_elements * sizeof(cuFloatComplex));
    cudaMalloc((void**)&d_out, num_elements * sizeof(cuFloatComplex));

    ReadMatlabBin(InputPath,h_in,rows,cols);
    
    int target_row = 1;
    int target_col = 1;

    // 拷贝数据到设备
    cudaMemcpy(d_in, h_in, num_elements * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    

    
    // 调用 CFAR 内核
    PROFILE_START(cfar);
    launch_ca_cfar(d_in, d_out, rows, cols, guard_rows, guard_cols, train_rows, train_cols, threshold_scale);
    PROFILE_END(cfar, "CFAR");


    // 将结果从设备拷贝回主机
    cudaMemcpy(h_out, d_out, num_elements * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    WriteMatlabBin(outputPath,h_out,rows,cols);


    // 释放资源
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
