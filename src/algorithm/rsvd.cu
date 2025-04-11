#include "rsvd.h"
#include "tools.h"

// 核函数：创建单位矩阵（列主序）
__global__ void setIdentity(cuFloatComplex* mat, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size && col < size) {
        mat[row + col * size] = make_cuFloatComplex((row == col) ? 1.0f : 0.0f, 0.0f);
    }
}


void de_clutter_rsvd(cuFloatComplex* d_in, cuFloatComplex* d_out, int m, int n, int N, int k, int p, int niters) {
    // 初始化 CUDA 
    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;
    cudaStream_t stream = nullptr;
    cusolverDnParams_t params = nullptr;

    cusolverDnCreate(&cusolverH);
    cusolverDnCreateParams(&params);
    cublasCreate(&cublasH);
    cudaStreamCreate(&stream);
    cusolverDnSetStream(cusolverH, stream);
    cublasSetStream(cublasH, stream);

    // 统一分配CUDA内存
    cuFloatComplex* d_fc;
    cudaMalloc(&d_fc, sizeof(cuFloatComplex) * (4 * m * m + 3 * m * k));
    cuFloatComplex* d_Rc = d_fc;
    cuFloatComplex* d_U = d_Rc + m * m;
    cuFloatComplex* d_Vt = d_U + m * k;
    cuFloatComplex* d_Vh = d_Vt + k * m;
    cuFloatComplex* d_Pc = d_Vh + k * m;
    cudaMemset(d_Pc, 0, m * m * sizeof(cuFloatComplex));
    cuFloatComplex* d_P = d_Pc + m * m;
    cuFloatComplex* d_I = d_P + m * m;

    // 1. 计算 Rc = Y*Y'（m x m）
    cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
    cuFloatComplex beta = make_cuFloatComplex(0.0f, 0.0f);
    cublasCgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, m, m, n, &alpha, d_in, m, d_in, m, &beta, d_Rc, m);

    // 2. 执行rSVD分解
    float* d_S;
    cudaMalloc(&d_S, k * sizeof(float));

    size_t d_ws_size, h_ws_size;
    //PROFILE_START(cusolverDnXgesvdr_func);
    cusolverDnXgesvdr_bufferSize(cusolverH, params, 'S', 'S', m, m, k, p, niters, CUDA_C_32F, d_Rc, m, CUDA_R_32F, d_S, CUDA_C_32F, d_U, m, CUDA_C_32F, d_Vt, m, CUDA_C_32F, &d_ws_size, &h_ws_size);

    void* d_work;
    cudaMalloc(&d_work, d_ws_size);
    void* h_work = malloc(h_ws_size);
    int* d_info;
    cudaMalloc(&d_info, sizeof(int));

    cusolverDnXgesvdr(cusolverH, params, 'S', 'S', m, m, k, p, niters, CUDA_C_32F, d_Rc, m, CUDA_R_32F, d_S, CUDA_C_32F, d_U, m, CUDA_C_32F, d_Vt, m, CUDA_C_32F, d_work, d_ws_size, h_work, h_ws_size, d_info);
    //PROFILE_END(cusolverDnXgesvdr_func, "    rsvd分解");
    // 3. 使用 cublasCgeam 进行共轭转置（V^H = conj(V)^T）
    //PROFILE_START(cublasCgeam_func);
    cublasCgeam(
        cublasH,
        CUBLAS_OP_C,  // 共轭转置 (conj-transpose)
        CUBLAS_OP_N,  // 不转置
        k,  // V^H 的行数
        m,  // V^H 的列数
        &alpha,  // alpha = 1
        d_Vt,  // 输入的 V (m×k)
        m,  // lda = V 的 leading dimension
        &beta,  // beta = 0
        NULL,  // 不使用的矩阵
        k,  // ldb (不适用)
        d_Vh,  // 输出的 V^H (k×m)
        k  // ldc = V^H 的 leading dimension
    );
    //PROFILE_END(cublasCgeam_func, "    计算共轭转置");
    // 4. 计算 Pc = sum(U(:,1:N)*V(:,1:N)')（m x m）
    //PROFILE_START(sum);
    for (int i = 0; i < N; ++i) {
        // U的第i列 (m x 1) 与 Vt的第i行 (1 x n) 的外积（需要调整内存访问）
        cublasCgeru(
            cublasH, m, m, &alpha, d_U + i * m, 1,  // U的第i列
            d_Vh + i * k, k,  // Vt的第i行（跨度为k）
            d_Pc, m);
    }
    //PROFILE_END(sum, "    计算Pc");

    // 5. 计算 P = I - Pc*Pc'（m x m）
    //PROFILE_START(P);
    // 计算 Pc*Pc'
    cublasCgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, m, m, m, &alpha, d_Pc, m, d_Pc, m, &beta, d_P, m);

    // 创建单位矩阵
    dim3 block(16, 16);
    dim3 grid((m + 15) / 16, (m + 15) / 16);
    setIdentity<<<grid, block>>>(d_I, m);

    // P = I - Pc*Pc'
    cuFloatComplex neg_alpha = make_cuFloatComplex(-1.0f, 0.0f);
    cublasCaxpy(cublasH, m * m, &neg_alpha, d_P, 1, d_I, 1);
    //PROFILE_END(P, "    计算P");
    // 6. 计算 X = P*Y（m x n）
    //PROFILE_START(X);
    cublasCgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m, &alpha, d_I, m, d_in, m, &beta, d_out, m);
    //PROFILE_END(X, "    计算X");
    // 清理资源
    cudaFree(d_fc);
    cudaFree(d_work);
    free(h_work);
    cudaFree(d_info);
    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);
    cudaStreamDestroy(stream);
}


