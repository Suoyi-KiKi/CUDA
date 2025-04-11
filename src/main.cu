#include "common_headers.h"
#include "dataProcess.h"
#include "cfar.h"
#include "tools.h"
#include "mtd.h"
#include "rsvd.h"


int main() {
/*-------------------------------------------------分配内存-------------------------------------------------------------- */
    int rows = 100;
    int cols = 100;
    int num_elements = rows * cols;

    // 配置 CFAR 参数（可自行调整）
    int guard_rows = 15, guard_cols = 5;
    int train_rows = 15, train_cols = 10;
    // float threshold_scale = 1.4f;
    float pfa = pow(10.0f, -6.5f);
    float threshold_scale = (pow(pfa, -1.0f / 10920.0f) - 1.0f) * 10920.0f;

    const std::string InputPath = "/home/kiki/MYCUDA/Bin/ramdon.bin";
    const std::string mtdoutputPath = "/home/kiki/MYCUDA/Bin/MTDOUT.bin";
    const std::string rsvdoutputPath = "/home/kiki/MYCUDA/Bin/RSVDOUT.bin";
    const std::string cfaroutputPath = "/home/kiki/MYCUDA/Bin/CFAROUT.bin";

    cuFloatComplex* h_in =  (cuFloatComplex*) malloc(num_elements * sizeof(cuFloatComplex));
    cuFloatComplex* h_out = (cuFloatComplex*) malloc(num_elements * sizeof(cuFloatComplex));
    cuFloatComplex* d_in ;
    cuFloatComplex* d_out;
    cuFloatComplex* temp;
    cudaMalloc((void**)&d_in, num_elements * sizeof(cuFloatComplex));
    cudaMalloc((void**)&d_out, num_elements * sizeof(cuFloatComplex));

    ReadMatlabBin(InputPath,h_in,rows,cols);
    cudaMemcpy(d_in, h_in, num_elements * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);


/*-------------------------------------------------rsvd-------------------------------------------------------------- */
    PROFILE_START(rsvd);
    de_clutter_rsvd(d_in, d_out, rows, cols, 1, 10, 2, 3);
    PROFILE_END(rsvd, "RSVD");

/*-------------------------------------------------mtd-------------------------------------------------------------- */

    PROFILE_START(mtd);
    temp = d_in;
    d_in = d_out;
    d_out = temp;
    MTD_CUDA_SIM_2D_C_Style(d_in,d_out,rows,cols);
    PROFILE_END(mtd, "MTD");

/*-------------------------------------------------cfar-------------------------------------------------------------- */

    PROFILE_START(cfar);
    temp = d_in;
    d_in = d_out;
    d_out = temp;
    launch_ca_cfar(d_in, d_out, rows, cols, guard_rows, guard_cols, train_rows, train_cols, threshold_scale);
    PROFILE_END(cfar, "cfar");

/*-------------------------------------------------释放内存-------------------------------------------------------------- */

    cudaMemcpy(h_out, d_out, num_elements * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    WriteMatlabBin(cfaroutputPath,h_out,rows,cols);

    
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
