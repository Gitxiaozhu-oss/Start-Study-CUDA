#include <iostream>
#include <cuda_runtime.h>

// 定义一个 CUDA 核函数，用于实现向量加法
// 参数 A 和 B 是输入向量，C 是输出向量，N 是向量的长度
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    // 计算当前线程对应的向量索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 检查索引是否在向量长度范围内
    if (i < N) {
        // 执行向量加法操作
        C[i] = A[i] + B[i];
    }
}

int main() {
    // 定义向量的长度
    int N = 1024;
    // 计算向量所需的内存大小
    size_t size = N * sizeof(float);

    // 在主机（CPU）上分配内存，用于存储输入和输出向量
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // 初始化输入向量 A 和 B
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 定义指向设备（GPU）内存的指针
    float *d_A, *d_B, *d_C;
    // 在设备上为向量 A、B 和 C 分配内存
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 将主机上的输入向量 A 和 B 复制到设备上
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义每个线程块中的线程数量
    int threadsPerBlock = 256;
    // 计算所需的线程块数量
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 正确调用 CUDA 核函数，指定网格和线程块的维度
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 将设备上的输出向量 C 复制到主机上
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 输出输出向量 C 的前 10 个元素
    for (int i = 0; i < 10; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // 释放主机上分配的内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    // 释放设备上分配的内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}