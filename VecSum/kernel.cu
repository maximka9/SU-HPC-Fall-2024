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
    cudaMalloc(&d_vector, size * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_vector, vector, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    sumVectorGPUKernel << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int) >> > (d_vector, d_result, size);
    cudaDeviceSynchronize();
    cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_vector);
    cudaFree(d_result);
}
__global__ void sumVectorGPUReductionKernel(int* input, int* output, int n) {
    extern __shared__ int cache[];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // Загружаем данные в shared memory
    if (index < n) cache[threadIdx.x] = input[index];
    else  cache[threadIdx.x] = 0;  // Если индекс за пределами массива, заполняем нулями
    __syncthreads();

    // Редукция внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)  cache[threadIdx.x] += cache[threadIdx.x + stride];
        
        __syncthreads();
    }
    // Сохраняем результат каждого блока в глобальную память
    if (threadIdx.x == 0) {
        output[blockIdx.x] = cache[0];
    }
}

// Финальная редукция для всех блоков
__global__ void finalizeReductionKernel(int* output, int* result, int n) {
    extern __shared__ int cache[];
    int index = threadIdx.x;
    // Загружаем частичные суммы в shared memory
    if (index < n) cache[index] = output[index];
    else cache[index] = 0;  // Заполняем нулями, если поток вне допустимого диапазона
    __syncthreads();
    // Редукция внутри блока (только один блок)
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (index < stride) {
            cache[index] += cache[index + stride];
        }
        __syncthreads();
    }

    // Сохраняем итоговый результат
    if (index == 0)  *result = cache[0];
    
}

void sumVectorGPUReduction(const int* vector, int* result, int size) {
    int* d_vector;
    int* d_partial_sums;
    int* d_result;
    int h_result = 0;

    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Выделяем память на GPU
    cudaMalloc(&d_vector, size * sizeof(int));
    cudaMalloc(&d_partial_sums, blocksPerGrid * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_vector, vector, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    // Запускаем редукцию по блокам
    sumVectorGPUReductionKernel << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int) >> > (d_vector, d_partial_sums, size);
    cudaDeviceSynchronize();

    // Если больше одного блока, запускаем финальную редукцию
    if (blocksPerGrid > 1) {
        finalizeReductionKernel << <1, threadsPerBlock, threadsPerBlock * sizeof(int) >> > (d_partial_sums, d_result, blocksPerGrid);
        cudaDeviceSynchronize();
    }
    else {
        cudaMemcpy(result, d_partial_sums, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_vector);
    cudaFree(d_partial_sums);
    cudaFree(d_result);
}
