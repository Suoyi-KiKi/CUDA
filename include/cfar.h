#ifndef _CFAR_H_
#define _CFAR_H_


#include "common_headers.h"

// 主机接口函数：配置线程块、共享内存并启动内核
void launch_ca_cfar(const cuFloatComplex* d_in, cuFloatComplex* d_out,
    int rows, int cols,
    int guard_rows, int guard_cols,
    int train_rows, int train_cols,
    float threshold_scale);


#endif