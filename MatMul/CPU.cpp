#include <iostream>
#include <locale>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream> 
// Объявление CUDA-функции
void multiplyMatricesGPU(const int* A, const int* B, int* C, int n);

// Функция для последовательного умножения матриц на CPU
void multiplyMatricesCPUSequential(const int* A, const int* B, int* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Функция для генерации случайных матриц
void generateRandomMatrix(int* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = rand() % 10;
    }
}

// Функция для проверки корректности
bool verify(const int* A, const int* B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (A[i * n + j] != B[i * n + j]) return false;
        }
    }
    return true;
}

int main() {
    setlocale(LC_ALL, "Russian");

    const int num_tests = 5; // Количество тестов
    const int start_n = 256;   // Начальное значение n
    const int step = 128;       // Шаг
    int n_values[num_tests]; 
   
    for (int i = 0; i < num_tests; i++) {
        n_values[i] = start_n + i * step; 
    }
    srand(static_cast<unsigned>(time(0)));

    // Открытие файла для записи результатов
    std::ofstream results("results.csv");
    results << "n,CPU_Time(s),GPU_Time(s),Speedup\n";

    // Перебор значений n
    for (int i = 0; i < num_tests; i++) {
        int n = n_values[i];
        int* A = new int[n * n];
        int* B = new int[n * n];
        int* C_CPU_Seq = new int[n * n];
        int* C_GPU = new int[n * n];

        // Генерация случайных значений для матриц
        generateRandomMatrix(A, n);
        generateRandomMatrix(B, n);

        // Умножение на CPU
        auto start_cpu_seq = std::chrono::high_resolution_clock::now();
        multiplyMatricesCPUSequential(A, B, C_CPU_Seq, n);
        auto end_cpu_seq = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_cpu_seq = end_cpu_seq - start_cpu_seq;

        // Умножение на GPU
        auto start_gpu = std::chrono::high_resolution_clock::now();
        multiplyMatricesGPU(A, B, C_GPU, n);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_gpu = end_gpu - start_gpu;

        double speedup = duration_cpu_seq.count() / duration_gpu.count();

        // Запись результатов в файл
        results << n << ","
            << duration_cpu_seq.count() << ","
            << duration_gpu.count() << ","
            << speedup << "\n";

        // Проверка корректности
        if (verify(C_CPU_Seq, C_GPU, n)) {
             std::cout << "\nРезультаты на CPU и GPU совпадают для n = " << n << "!" << std::endl;
        }
        else {
            std::cout << "\nРезультаты на CPU и GPU не совпадают для n = " << n << "!" << std::endl;
        }

        delete[] A;
        delete[] B;
        delete[] C_CPU_Seq;
        delete[] C_GPU;
    }
    results.close();
    return 0;
}
