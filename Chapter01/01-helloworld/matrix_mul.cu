#include <iostream>
#include <cuda_runtime.h>

// 定义每个线程块处理的矩阵块的宽度
#define TILE_WIDTH 16

// CUDA 核函数，用于执行矩阵乘法
__global__ void matrixMul(const float *A, const float *B, float *C, int width) {
    // 定义共享内存数组，用于存储矩阵 A 和 B 的子块
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    // 获取当前线程块的索引
    int bx = blockIdx.x, by = blockIdx.y;
    // 获取当前线程在块内的索引
    int tx = threadIdx.x, ty = threadIdx.y;

    // 计算当前线程处理的元素在结果矩阵 C 中的行和列索引
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // 初始化累加器，用于存储当前元素的计算结果
    float sum = 0.0f;

    // 分块计算矩阵乘法，遍历所有的子块
    for (int m = 0; m < width / TILE_WIDTH; m++) {
        // 将矩阵 A 和 B 的子块从全局内存加载到共享内存
        s_A[ty][tx] = A[row * width + (m * TILE_WIDTH + tx)];
        s_B[ty][tx] = B[(m * TILE_WIDTH + ty) * width + col];
        // 同步线程，确保所有线程都完成了数据加载
        __syncthreads();

        // 在共享内存中进行矩阵乘法的部分计算
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        // 同步线程，确保所有线程都完成了部分计算
        __syncthreads();
    }

    // 检查计算结果的索引是否在矩阵范围内
    if (row < width && col < width) {
        // 将计算结果存储到全局内存中的结果矩阵 C 中
        C[row * width + col] = sum;
    }
}

int main() {
    // 定义矩阵的宽度
    int width = 64;
    // 计算矩阵所需的内存大小
    size_t size = width * width * sizeof(float);

    // 在主机（CPU）上分配内存，用于存储矩阵 A、B 和结果矩阵 C
    float *h_A = new float[width * width];
    float *h_B = new float[width * width];
    float *h_C = new float[width * width];

    // 初始化矩阵 A 和 B 的元素
    for (int i = 0; i < width * width; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 定义指向设备（GPU）内存的指针
    float *d_A, *d_B, *d_C;
    // 在设备上为矩阵 A、B 和结果矩阵 C 分配内存
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 将主机上的矩阵 A 和 B 复制到设备上
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义每个线程块的维度
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    // 定义网格的维度
    dim3 blocksPerGrid(width / TILE_WIDTH, width / TILE_WIDTH);

    // 正确调用 CUDA 核函数，指定网格和线程块的维度
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);

    // 将设备上的结果矩阵 C 复制到主机上
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 输出结果矩阵 C 的前 10 个元素
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