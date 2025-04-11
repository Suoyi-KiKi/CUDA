#include "dataProcess.h"


void ReadMatlabBin(const std::string& filename, cuFloatComplex* data, int rows, int cols)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    // 计算预期文件大小（复数数量 x 2个float）
    const size_t expected_size = 2 * rows * cols * sizeof(float);
    in.seekg(0, std::ios::end);
    size_t actual_size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (actual_size != expected_size) {
        std::cerr << "Error: File size mismatch. Expected " << expected_size
        << " bytes, got " << actual_size << " bytes." << std::endl;
        exit(1);
    }

    // 读取交替存储的实部虚部数据
    std::vector<float> interleaved_data(2 * rows * cols);
    in.read(reinterpret_cast<char*>(interleaved_data.data()), actual_size);

    // 重建复数数组（直接填充用户提供的指针）
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = make_cuFloatComplex(
            interleaved_data[2*i],      // 实部
            interleaved_data[2*i + 1]   // 虚部
        );
    }

    // 验证输出
    std::cout << "BIn文件前三个读取数据为:\n";
    for (int i = 0; i < 3; ++i) {
      printf("    [%d] (%.6f, %.6f)\n", i, data[i].x, data[i].y);
    }
}


void WriteMatlabBin(const std::string& filename,  const cuFloatComplex* data,  int rows, int cols) 
{
    // 新增指针有效性验证
    if (data == nullptr) {
        std::cerr << "Error: Data pointer is null!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // 通过维度计算数据量
    const size_t num_elements = rows * cols;

    // 维度合理性验证（替换原vector大小检查）
    if (rows <= 0 || cols <= 0) {
        std::cerr << "Error: Invalid dimensions. rows: " << rows
                 << ", cols: " << cols << std::endl;
        exit(EXIT_FAILURE);
    }

    // 打开文件（保持原有逻辑）
    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot create output file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // 构建交错数据（直接操作指针）
    std::vector<float> interleaved(2 * num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        interleaved[2*i] = data[i].x;    // 实部
        interleaved[2*i + 1] = data[i].y; // 虚部
    }

    // 写入文件（保持原有逻辑）
    out.write(reinterpret_cast<const char*>(interleaved.data()),
             interleaved.size() * sizeof(float));

    // 检查写入状态（保持原有逻辑）
    if (!out.good()) {
        std::cerr << "Error: Failed during file writing!" << std::endl;
        exit(EXIT_FAILURE);
    }
    out.close();
}



/**
 * @brief 将cuFloatComplex数组转换为Eigen Tensor
 * @param input 输入数组
 * @param output 输出Eigen Tensor
 * @param rows 行数
 * @param cols 列数
 * @param depth 深度
 */
void convert_to_eigen_tensor(const cuFloatComplex* input, Eigen::Tensor<thrust::complex<float>, 3, Eigen::ColMajor>& output, int rows, int cols, int depth) {
    output = Eigen::Tensor<thrust::complex<float>, 3, Eigen::ColMajor>(rows, cols, depth);
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int index = d * rows * cols + r * cols + c;
                thrust::complex<float> value(input[index].x, input[index].y);
                output(r, c, d) = value;
            }
        }
    }
}

// 高效转换内核（向量化，零额外内存分配）
__global__ void convertHalfToFloat_Optimized(__half* in, cuFloatComplex* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 一次加载 2 个 __half（实部和虚部）
        __half2 val = *reinterpret_cast<__half2*>(&in[2 * idx]);
        // 转换为 float2
        float2 f_val = __half22float2(val);
        // 写入 cuFloatComplex
        out[idx] = make_cuFloatComplex(f_val.x, f_val.y);
    }
}


// 读取复数向量的函数
Eigen::VectorXcf readComplexVectorFromBin(const std::string& filePath, int size) {
    Eigen::VectorXcf vec(size);
    std::ifstream file(filePath, std::ios::binary);

    if (!file) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }

    // 读取实部
    for (int i = 0; i < size; ++i) {
        float realPart;
        file.read(reinterpret_cast<char*>(&realPart), sizeof(float));
        vec(i).real(realPart);
    }

    // 读取虚部
    for (int i = 0; i < size; ++i) {
        float imagPart;
        file.read(reinterpret_cast<char*>(&imagPart), sizeof(float));
        vec(i).imag(imagPart);
    }

    file.close();
    return vec;
}

void saveTensorToBinaryFile(const Eigen::Tensor<Complex, 3>& tensor, const std::string& filePath) {
    // 打开二进制文件用于写入
    std::ofstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filePath << std::endl;
        return;
    }

    // 获取矩阵的维度
    int M = tensor.dimension(0);
    int N = tensor.dimension(1);
    int P = tensor.dimension(2);

    // 计算总元素个数
    int numElements = M * N * P;

    // 读取实部和虚部到临时数组
    std::vector<float> realPart(numElements);
    std::vector<float> imagPart(numElements);

    // 分别提取实部和虚部
    for (int i = 0; i < numElements; ++i) {
        realPart[i] = tensor.data()[i].real();
        imagPart[i] = tensor.data()[i].imag();
    }

    // 写入实部
    file.write(reinterpret_cast<const char*>(realPart.data()), numElements * sizeof(float));

    // 写入虚部
    file.write(reinterpret_cast<const char*>(imagPart.data()), numElements * sizeof(float));

    // 关闭文件
    file.close();
}