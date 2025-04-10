#include "common_headers.h"
#include "dataProcess.h"
#include "mtd.h"
#include "tools.h"

int main(void){

    int rows = 100;
    int cols = 100;
    int num_elements = rows * cols;

    const std::string InputPath = "/home/kiki/MYCUDA/Bin/ramdon.bin";
    const std::string outputPath = "/home/kiki/MYCUDA/Bin/MTDOUT.bin";

    cuFloatComplex* h_in =  (cuFloatComplex*) malloc(num_elements * sizeof(cuFloatComplex));
    cuFloatComplex* h_out = (cuFloatComplex*) malloc(num_elements * sizeof(cuFloatComplex));
    cuFloatComplex* d_in;
    cuFloatComplex* d_out;
    cudaMalloc((void**)&d_in, num_elements * sizeof(cuFloatComplex));
    cudaMalloc((void**)&d_out, num_elements * sizeof(cuFloatComplex));

    ReadMatlabBin(InputPath,h_in,rows,cols);

    cudaMemcpy(d_in, h_in, num_elements * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    PROFILE_START(MTD);
    MTD_CUDA_SIM_2D_C_Style(d_in,d_out,rows,cols);
    PROFILE_END(MTD, "MTD");


    cudaMemcpy(h_out, d_out, num_elements * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    WriteMatlabBin(outputPath,h_out,rows,cols);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}