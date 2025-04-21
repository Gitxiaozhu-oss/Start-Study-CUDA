#include<stdio.h>
#include<stdlib.h> 

// CUDA 核函数
__global__ void say_hi() {
    printf("Hello from GPU thread (%d, %d)\n", threadIdx.x, blockIdx.x);
}

int main() {
	printf("Hello World from host!\n"); 
	 // 调用 CUDA 核函数
    say_hi<<<1,1>>>();
	
	// 同步设备，确保核函数执行完毕
    cudaDeviceSynchronize();
	printf("Hello from CPU endl\n");
    return 0;
}