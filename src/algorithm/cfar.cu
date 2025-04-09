#include "cfar.h"

// 内联函数：计算复数幅值的平方
__device__ __forceinline__ float norm2f(const cuFloatComplex val) {
    return val.x * val.x + val.y * val.y;
}

// 2D CA-CFAR 内核
__global__ void ca_cfar_2d_kernel(const cuFloatComplex* __restrict__ in,
                                  cuFloatComplex* __restrict__ out,
                                  int nrows, int ncols,
                                  int guard_rows, int guard_cols,
                                  int train_rows, int train_cols,
                                  float threshold_scale)
{
    // 预计算窗口半径：保护+训练单元
    const int half_win_x = guard_cols + train_cols;
    const int half_win_y = guard_rows + train_rows;
    
    // 当前线程对应的输出位置（CUT 单元）
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 共享内存区域尺寸（覆盖整个线程块 + 边界扩展）
    const int shared_w = blockDim.x + 2 * half_win_x;
    const int shared_h = blockDim.y + 2 * half_win_y;
    
    // 声明动态共享内存（以 cuFloatComplex 为单位）
    extern __shared__ cuFloatComplex s_data[];
    
    // 计算 tile 在全局内存中的起始位置
    const int tile_x0 = blockIdx.x * blockDim.x;
    const int tile_y0 = blockIdx.y * blockDim.y;
    
    // 每个线程循环加载共享内存（避免线程 id 数量不足）
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_shared = shared_w * shared_h;
    const int local_threads = blockDim.x * blockDim.y;
    for (int idx = thread_id; idx < total_shared; idx += local_threads) {
        int s_y = idx / shared_w;
        int s_x = idx - s_y * shared_w;
        int global_x = tile_x0 - half_win_x + s_x;
        int global_y = tile_y0 - half_win_y + s_y;
        if (global_x < 0 || global_x >= ncols || global_y < 0 || global_y >= nrows) {
            s_data[idx] = make_cuFloatComplex(0.0f, 0.0f);
        } else {
            s_data[idx] = in[global_y * ncols + global_x];
        }
    }
    __syncthreads();
    
    // 若当前 CUT 超出矩阵有效区域，则退出
    if (out_x >= ncols || out_y >= nrows)
        return;
    
    // 当前 CUT 在共享内存中的索引（中心）
    const int center_sx = threadIdx.x + half_win_x;
    const int center_sy = threadIdx.y + half_win_y;
    
    // 累计训练单元（非保护区域）的噪声功率
    float noise_sum = 0.0f;
    int train_count = 0;
    for (int dy = -half_win_y; dy <= half_win_y; ++dy) {
        for (int dx = -half_win_x; dx <= half_win_x; ++dx) {
            // 跳过保护区域（包括 CUT 本身）
            if (abs(dx) <= guard_cols && abs(dy) <= guard_rows) continue;
            int s_idx = (center_sy + dy) * shared_w + (center_sx + dx);
            noise_sum += norm2f(s_data[s_idx]);
            train_count++;
        }
    }
    
    // 防止除以0：当训练单元计数为 0 时，设置为最大浮点值
    float noise_avg = (train_count > 0) ? (noise_sum / train_count) : FLT_MAX;
    float threshold = noise_avg * threshold_scale;
    
    // 计算 CUT 自身功率
    cuFloatComplex cell_val = s_data[center_sy * shared_w + center_sx];
    float cell_power = norm2f(cell_val);
    
    // 判断：若 CUT 功率大于阈值，则检测为目标，输出原始信号；否则输出零
    int detect = (cell_power > threshold) ? 1 : 0;
    out[out_y * ncols + out_x] = detect ? cell_val : make_cuFloatComplex(0.0f, 0.0f);
}

// 主机接口函数：配置线程块、共享内存并启动内核
void launch_ca_cfar(const cuFloatComplex* d_in, cuFloatComplex* d_out,
                    int rows, int cols,
                    int guard_rows, int guard_cols,
                    int train_rows, int train_cols,
                    float threshold_scale)
{
    // 建议的线程块配置
    cudaStream_t stream = 0;
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    
    // 计算共享内存大小
    int half_win_x = guard_cols + train_cols;
    int half_win_y = guard_rows + train_rows;
    int shared_mem_size = (block.x + 2 * half_win_x) * (block.y + 2 * half_win_y) * sizeof(cuFloatComplex);
    
    // 检查共享内存大小是否超过设备允许的最大值
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    assert(shared_mem_size <= prop.sharedMemPerBlock);
    
    // 启动内核
    ca_cfar_2d_kernel<<<grid, block, shared_mem_size, stream>>>(d_in, d_out,
                                                                 rows, cols,
                                                                 guard_rows, guard_cols,
                                                                 train_rows, train_cols,
                                                                 threshold_scale);
    
    // 检查内核启动是否有错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
