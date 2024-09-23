#include <iostream>
#include <cuda_runtime.h>

// CUDA-ядро для умножения матриц на GPU
__global__ void multiplyMatricesGPUKernel(const int* A, const int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Функция для выполнения умножения матриц на GPU
void multiplyMatricesGPU(const int* A, const int* B, int* C, int n) {
    // Выделение памяти на GPU
    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, n * n * sizeof(int));
    cudaMalloc(&d_B, n * n * sizeof(int));
    cudaMalloc(&d_C, n * n * sizeof(int));

    // Копирование данных с CPU на GPU
    cudaMemcpy(d_A, A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Определение размеров блоков и сетки
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Запуск CUDA-ядра
    multiplyMatricesGPUKernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Копирование результата обратно на CPU
    cudaMemcpy(C, d_C, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
