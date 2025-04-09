#include "common_headers.h"
#include "dataProcess.h"
#include "rsvd.h"
#include "tools.h"

int main(void){

    int rows = 500;
    int cols = 3*800;
    int num_elements = rows * cols;

    const std::string InputPath = "/home/kiki/MYCUDA/Bin/INPUT.bin";
    const std::string outputPath = "/home/kiki/MYCUDA/Bin/RSVDOUT.bin";

    cuFloatComplex* h_in =  (cuFloatComplex*) malloc(num_elements * sizeof(cuFloatComplex));
    cuFloatComplex* h_out = (cuFloatComplex*) malloc(num_elements * sizeof(cuFloatComplex));
    cuFloatComplex* d_in;
    cuFloatComplex* d_out;
    cudaMalloc((void**)&d_in, num_elements * sizeof(cuFloatComplex));
    cudaMalloc((void**)&d_out, num_elements * sizeof(cuFloatComplex));

    ReadMatlabBin(InputPath,h_in,rows,cols);

    cudaMemcpy(d_in, h_in, num_elements * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    PROFILE_START(rsvd);
    de_clutter_rsvd(d_in, d_out, rows, cols, 1, 10, 2, 2);
    PROFILE_END(rsvd, "RSVD");


    cudaMemcpy(h_out, d_out, num_elements * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    WriteMatlabBin(outputPath,h_out,rows,cols);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}