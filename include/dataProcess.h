//所有对bin文件进行的操作
//所有对数据的操作

#ifndef _DATAPROCESS_H_
#define _DATAPROCESS_H_

#include "common_headers.h"

typedef thrust::complex<float> Complex;  // 单精度复数类型


//从bin文件中读取二维复数矩阵
void ReadMatlabBin(const std::string& filename, cuFloatComplex* data, int rows, int cols);
//将二维复数矩阵写入一个bin文件
void WriteMatlabBin(const std::string& filename, const cuFloatComplex* data, int rows,  int cols) ;
//将cuFloatComplex数组转换为Eigen Tensor
void convert_to_eigen_tensor(const cuFloatComplex* input, Eigen::Tensor<thrust::complex<float>, 3, Eigen::ColMajor>& output, int rows, int cols, int depth);
//将半精度输入转化成单精度输出
__global__ void convertHalfToFloat_Optimized(__half* in, cuFloatComplex* out, int n);
//读bin文件返回eigen
Eigen::VectorXcf readComplexVectorFromBin(const std::string& filePath, int size);
//将eigen写入到bin
void saveTensorToBinaryFile(const Eigen::Tensor<Complex, 3>& tensor, const std::string& filePath);


#endif