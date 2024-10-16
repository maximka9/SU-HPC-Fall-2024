#include <iostream>
#include <locale>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <thread>
#include <vector>
#include <omp.h>

// Объявление CUDA-функций для суммирования элементов вектора на GPU
void sumVectorGPU(const int* vector, int* result, int size);
void sumVectorGPUReduction(const int* vector, int* result, int size);

// Функция для последовательного суммирования элементов вектора на CPU
int sumVectorCPU(const int* vector, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += vector[i];
    }
    return sum;
}

// Многопоточная функция для суммирования вектора на CPU с использованием 2 потоков
int sumVectorCPUMultithreaded(const int* vector, int size) {
    int total_sum = 0;

#pragma omp parallel for reduction(+:total_sum) num_threads(2)
    for (int i = 0; i < size; i++) {
        total_sum += vector[i];
    }
    return total_sum;
}

void generateRandomVector(int* vector, int size) {
    for (int i = 0; i < size; i++) {
       // vector[i] = rand() % 10;  
        vector[i] = 1;
    }
}
int main() {
    setlocale(LC_ALL, "Russian");

    const int initial_size = 0;  // Начальный размер вектора
    const int step_size = 100000; // Шаг увеличения
    const int num_tests = 11;     // Количество тестов
    const int max_size = initial_size + (num_tests - 1) * step_size;

    srand(static_cast<unsigned>(time(0)));
    std::ofstream results("results.csv");
    results << "Vector Size, CPU Time (s), CPU Multithreaded Time (s), GPU Atomic Time (s), GPU Reduction Time (s), Speedup CPU/Multithreaded, Speedup CPU/GPU Atomic, Speedup CPU/GPU Reduction\n";

    for (int test = 0; test < num_tests; ++test) {
        int vector_size = initial_size + test * step_size;
        int* vector = new int[vector_size];
        generateRandomVector(vector, vector_size);
        int cpu_result = 0;
        int cpu_multithreaded_result = 0;
        int gpu_atomic_result = 0;
        int gpu_reduction_result = 0;

        // Суммирование на CPU (последовательное)
        auto start_cpu = std::chrono::high_resolution_clock::now();
        cpu_result = sumVectorCPU(vector, vector_size);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;

        // Суммирование на CPU (многопоточное)
        auto start_cpu_mt = std::chrono::high_resolution_clock::now();
        cpu_multithreaded_result = sumVectorCPUMultithreaded(vector, vector_size);
        auto end_cpu_mt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_cpu_mt = end_cpu_mt - start_cpu_mt;

        // Суммирование на GPU (с использованием atomicAdd)
        auto start_gpu_atomic = std::chrono::high_resolution_clock::now();
        sumVectorGPU(vector, &gpu_atomic_result, vector_size);
        auto end_gpu_atomic = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_gpu_atomic = end_gpu_atomic - start_gpu_atomic;

        // Суммирование на GPU (с использованием редукции)
        auto start_gpu_reduction = std::chrono::high_resolution_clock::now();
        sumVectorGPUReduction(vector, &gpu_reduction_result, vector_size);
        auto end_gpu_reduction = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_gpu_reduction = end_gpu_reduction - start_gpu_reduction;

        if (vector_size > 1) {
            // Вычисление ускорения
            double speedup_cpu_multithreaded = duration_cpu.count() / duration_cpu_mt.count();
            double speedup_cpu_gpu_atomic = duration_cpu.count() / duration_gpu_atomic.count();
            double speedup_cpu_gpu_reduction = duration_cpu.count() / duration_gpu_reduction.count();

            results << vector_size << ","
                << duration_cpu.count() << ","
                << duration_cpu_mt.count() << ","
                << duration_gpu_atomic.count() << ","
                << duration_gpu_reduction.count() << ","
                << speedup_cpu_multithreaded << ","
                << speedup_cpu_gpu_atomic << ","
                << speedup_cpu_gpu_reduction << "\n";

            // Проверка корректности
            if (cpu_result == gpu_atomic_result && cpu_multithreaded_result == gpu_atomic_result && cpu_result == gpu_reduction_result) {
                // std::cout << "Результаты на CPU, многопоточном CPU, GPU (atomic) и GPU (reduction) совпадают! Сумма = " << cpu_result << std::endl;
            }
            else {
                std::cout << "Результаты на CPU, многопоточном CPU, GPU (atomic) и GPU (reduction) не совпадают!" << std::endl;
                std::cout << cpu_result << "  " << gpu_atomic_result << "  " << cpu_multithreaded_result << "  " << gpu_reduction_result << std::endl;
            }
        }
        delete[] vector;
    }

    results.close();
    return 0;
}
