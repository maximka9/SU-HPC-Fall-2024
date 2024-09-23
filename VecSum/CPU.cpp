#include <iostream>
#include <locale>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <thread>
#include <vector>
#include <omp.h>

// ���������� CUDA-������� ��� ������������ ��������� ������� �� GPU
void sumVectorGPU(const int* vector, int* result, int size);

// ������� ��� ����������������� ������������ ��������� ������� �� CPU
int sumVectorCPU(const int* vector, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += vector[i];
    }
    return sum;
}

// ������������� ������� ��� ������������ ������� �� CPU � �������������� 2 �������
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
        vector[i] = rand() % 10;  
    }
}
int main() {
    setlocale(LC_ALL, "Russian");

    const int initial_size = 0;  // ��������� ������ �������
    const int step_size = 200000;      // ��� ����������
    const int num_tests = 11;         // ���������� ������
    const int max_size = initial_size + (num_tests - 1) * step_size; 

    srand(static_cast<unsigned>(time(0)));
    std::ofstream results("results.csv");
    results << "Vector Size, CPU Time (s), CPU Multithreaded Time (s), GPU Time (s), Speedup CPU/Multithreaded, Speedup CPU/GPU\n";

    for (int test = 0; test < num_tests; ++test) {
        int vector_size = initial_size + test * step_size;
        int* vector = new int[vector_size];
        generateRandomVector(vector, vector_size);
        int cpu_result = 0;
        int cpu_multithreaded_result = 0;
        int gpu_result = 0;

        // ������������ �� CPU (����������������)
        auto start_cpu = std::chrono::high_resolution_clock::now();
        cpu_result = sumVectorCPU(vector, vector_size);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;

        // ������������ �� CPU (�������������)
        auto start_cpu_mt = std::chrono::high_resolution_clock::now();
        cpu_multithreaded_result = sumVectorCPUMultithreaded(vector, vector_size);
        auto end_cpu_mt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_cpu_mt = end_cpu_mt - start_cpu_mt;

        // ������������ �� GPU
        auto start_gpu = std::chrono::high_resolution_clock::now();
        sumVectorGPU(vector, &gpu_result, vector_size);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_gpu = end_gpu - start_gpu;
        if (vector_size > 1) {
            // ���������� ���������
            double speedup_cpu_multithreaded = duration_cpu.count() / duration_cpu_mt.count();
            double speedup_cpu_gpu = duration_cpu.count() / duration_gpu.count();
            
            results << vector_size << ","
                << duration_cpu.count()  << ","
                << duration_cpu_mt.count()  << ","
                << duration_gpu.count()  << ","
                << speedup_cpu_multithreaded << ","
                << speedup_cpu_gpu << "\n";

            // �������� ������������
            if (cpu_result == gpu_result && cpu_multithreaded_result == gpu_result) {
               // std::cout << "���������� �� CPU, ������������� CPU � GPU ���������! ����� = " << cpu_result << std::endl;
            }
            else {
                std::cout << "���������� �� CPU, ������������� CPU � GPU �� ���������!" << std::endl;
            }
        }
        delete[] vector;
    }

    results.close();
    return 0;
}


