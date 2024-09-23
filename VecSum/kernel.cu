#include <iostream>
#include <cuda_runtime.h>

// CUDA-ядро для суммирования элементов вектора на GPU
__global__ void sumVectorGPUKernel(int* d_data, int* d_result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        atomicAdd(d_result, d_data[index]);
    }
}

// Функция для выполнения суммирования элементов вектора на GPU
void sumVectorGPU(const int* vector, int* result, int size) {
    int* d_vector;
    int* d_result;
    int h_result = 0;

    // Выделение памяти на GPU
    cudaMalloc(&d_vector, size * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    // Копирование данных на GPU
    cudaMemcpy(d_vector, vector, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    // Определение размеров блоков и сетки
    int threadsPerBlock = 512; // Попробуйте 256 или 512
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Запуск CUDA-ядра с размером shared memory
    sumVectorGPUKernel << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int) >> > (d_vector, d_result, size);
    cudaDeviceSynchronize();

    // Копирование результата обратно на CPU
    cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Освобождение памяти
    cudaFree(d_vector);
    cudaFree(d_result);
}
