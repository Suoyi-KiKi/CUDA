#ifndef _RSVD_H_
#define _RSVD_H_

#include "common_headers.h"

/**
 * @brief 使用随机化SVD（Randomized SVD）方法对复矩阵进行低秩分解，适用于去噪或数据压缩（如杂波抑制）
 * @param d_in  设备端输入指针，指向待分解的复矩阵数据（大小为 m×n，按行优先存储）
 * @param d_out 设备端输出指针，用于存储分解后的结果。
 * @param m     输入矩阵的行数（原始信号维度，如距离单元数）
 * @param n     输入矩阵的列数（原始信号维度，如多普勒单元数）
 * @param N     原始信号的采样点数（可选参数，可能用于内部计算或校验）
 * @param k     目标秩（期望保留的主成分数量，k ≪ min(m, n)）
 * @param p     随机投影的过采样量（通常为5~10，用于提高数值稳定性）
 * @param niters 正交化迭代次数（如Power Iteration次数，默认为0或1，可提升小奇异值的精度）
 *
 * @note
 * 1. 函数假设输入矩阵在GPU设备内存中（d_in），输出结果直接写入设备内存（d_out）
 * 2. 随机化SVD通过随机投影近似计算奇异值分解，适合大规模矩阵的快速低秩近似
 * 3. 参数选择建议：
 *    - k: 根据实际需求设定（如保留能量占比≥95%）
 *    - p: 过采样量越大，精度越高但计算量增加（常用k+p ≤ min(m,n)）
 *    - niters: 对小奇异值敏感的场景可设为1~2
 * 4. 调用前需确保GPU内存已正确分配，且m/n/k等参数合法
 */
void de_clutter_rsvd(cuFloatComplex* d_in, cuFloatComplex* d_out, int m, int n, int N, int k, int p, int niters);


#endif