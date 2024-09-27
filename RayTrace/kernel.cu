#include <curand_kernel.h>   
#include <cuda_runtime.h>     
#include <vector_types.h>     
#include <vector_functions.h> 
#include <cmath>              
#include <vector>             
#include "EasyBMP.h"          
#include "geometry.h"         // операторы

// Функция для проверки пересечения луча с сферой
__device__ bool intersectSphere(const float3& rayOrigin, const float3& rayDir, const float3& sphereCenter, float radius, float& t) {
	float3 oc = rayOrigin - sphereCenter; // Вычисляем вектор от центра сферы до начала луча
	float a = dot(rayDir, rayDir);        // Вычисляем скалярное произведение направления луча
	float b = 2.0f * dot(oc, rayDir);     // Вычисляем удвоенное скалярное произведение
	float c = dot(oc, oc) - radius * radius; // Вычисляем расстояние от центра сферы до луча минус радиус в квадрате
	float discriminant = b * b - 4 * a * c;  // Вычисляем дискриминант уравнения

	if (discriminant < 0) return false;    // Если дискриминант отрицателен, пересечений нет
	t = (-b - sqrt(discriminant)) / (2.0f * a); // Вычисляем параметр t пересечения
	return t >= 0;                         // Возвращаем true, если пересечение существует
}
// Ядро для генерации случайных сфер
__global__ void generateRandomSpheres(Sphere* spheres, int numSpheres, unsigned long long seed) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; // Вычисляем индекс текущего потока
	if (idx < numSpheres) {
		curandState state;
		curand_init(seed, idx, 0, &state); // Инициализируем генератор случайных чисел для данного потока
		Sphere newSphere;

		newSphere.center.x = curand_uniform(&state) * 4 - 2.0f; // Генерация случайной координаты x в диапазоне [-2, 2]
		newSphere.center.y = 0.0f;                              // Устанавливаем y координату сферы на 0
		newSphere.center.z = curand_uniform(&state) * 4 - 2.5f; // Генерация случайной координаты z в диапазоне [-2.5, 1.5]
		newSphere.radius = 0.3f;                                // Устанавливаем фиксированный радиус сферы

		// Генерация случайных цветов
		newSphere.color.x = curand_uniform(&state); // Красный компонент цвета
		newSphere.color.y = curand_uniform(&state); // Зеленый компонент цвета
		newSphere.color.z = curand_uniform(&state); // Синий компонент цвета

		// Инициализация материала сферы
		newSphere.material.diffuse_color = newSphere.color; // Диффузный цвет совпадает с цветом сферы
		newSphere.material.specular_exponent = 50.0f;      // Устанавливаем экспоненту блеска (чем выше, тем ярче блеск)
		newSphere.material.albedo = make_float3(0.9f, 0.1f, 0.0f); // Определяем альбедо (соотношение диффузного и зеркального отражения)

		spheres[idx] = newSphere; // Сохраняем сгенерированную сферу в массив
	}
}
// Ядро для генерации случайных позиций источников света
__global__ void generateRandomLightPositions(float3* lights, int numLights, unsigned long long seed) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; // Вычисляем индекс текущего потока
	if (idx < numLights) {
		curandState state;
		curand_init(seed, idx, 0, &state); // Инициализация генератора случайных чисел для потока

		float x = curand_uniform(&state) * 2 - 1.0f; // Генерация случайной координаты x в диапазоне [-1, 1]
		float y = 1.5f;                             // Устанавливаем фиксированную координату y
		float z = 0.0f;                             // Устанавливаем фиксированную координату z
		printf("Light %d: x = %f, y = %f, z = %f\n", idx, x, y, z); // Вывод информации об источнике света
		lights[idx] = make_float3(x, y, z); // Заполняем массив позиций источников света
	}
}

__global__ void render(float3* output, int width, int height, Sphere* spheres, int numSpheres, float3* lights, int numLights) {
	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	if (x >= width || y >= height) return; 

	float3 rayOrigin = make_float3(0.0f, 0.0f, 2.0f); // позиция камеры
	float3 rayDir = make_float3((x - width / 2.0f) / (width / 2.0f), (height / 2.0f - y) / (height / 2.0f), -1.0f); // Направление луча
	rayDir = normalize(rayDir); // Нормализуем направление луча
	float3 hitColor = make_float3(1.0f, 1.0f, 1.0f); // Белый фон

	// Проверка пересечения с плоскостью y = -0.5
	float t_plane = (-0.3f - rayOrigin.y) / rayDir.y; // Вычисляем параметр t для пересечения с плоскостью
	if (t_plane >= 0) {
		float3 hitPoint = rayOrigin + rayDir * t_plane; // Вычисляем точку пересечения с плоскостью
		if (hitPoint.x >= -2.0f && hitPoint.x <= 2.0f) { // Проверяем, попадает ли точка в границы плоскости
			// Установите начальный цвет плоскости
			hitColor = make_float3(0.2f, 0.2f, 0.2f); // цвет плоскости в тени
			// Проверка на затенение
			for (int j = 0; j < numLights; j++) {
				float3 lightDir = normalize(lights[j] - hitPoint); // Направление от точки пересечения к источнику света
				float distanceToLight = length(lights[j] - hitPoint); // Расстояние до источника света
				// Проверяем, нет ли препятствий между источником света и точкой на плоскости
				bool isInShadow = false;
				for (int i = 0; i < numSpheres; i++) {
					float t;
					if (intersectSphere(hitPoint, lightDir, spheres[i].center, spheres[i].radius, t) && t < distanceToLight) {
						isInShadow = true;
						break;
					}
				}
				// Если точка не затенена, добавляем свет
				if (!isInShadow) {
					hitColor += make_float3(1.0f, 1.0f, 1.0f) * 0.5f;
				}
			}
		}
	}
	float t_min = 1e20f; // Инициализируем минимальное значение пересечения большим числом (для поиска ближайшего объекта)
	for (int i = 0; i < numSpheres; i++) { // Проходим по всем сферам
		float t;
		// Проверяем пересечение луча с i-й сферой и сохраняем t (параметр пересечения). Если пересечение есть и оно ближе (t < t_min), обновляем t_min
		if (intersectSphere(rayOrigin, rayDir, spheres[i].center, spheres[i].radius, t) && t < t_min) {
			t_min = t; // Обновляем минимальное значение пересечения
			float3 hitPoint = rayOrigin + rayDir * t; // Вычисляем точку пересечения луча со сферой
			float3 normal = normalize(hitPoint - spheres[i].center); // Вычисляем нормаль в точке пересечения (направление от центра сферы)

			float intensity = 0.0f; // Инициализация интенсивности света для текущей точки
			float attenuationFactor = 0.1f; // Коэффициент затухания света

			for (int j = 0; j < numLights; j++) {
				float3 lightDir = normalize(lights[j] - hitPoint); // Вычисляем направление света от источника к точке пересечения
				float distance = length(lights[j] - hitPoint); // Вычисляем расстояние от источника света до точки пересечения
				float attenuation = 1.0f / (1.0f + attenuationFactor * distance * distance); // Вычисляем коэффициент затухания света в зависимости от расстояния
				// Диффузное освещение
				// Увеличиваем интенсивность света на основе угла между нормалью и направлением света, умножая на коэффициент затухания
				intensity += fmaxf(0.0f, dot(normal, lightDir)) * attenuation;
				// Зеркальное освещение
				// Вычисляем направление отраженного луча
				float3 reflectDir = reflect(-lightDir, normal);
				// Увеличиваем интенсивность за счет зеркального отражения (на основе угла между отраженным лучом и направлением обзора), умножаем на коэффициент затухания
				intensity += powf(fmaxf(0.0f, dot(reflectDir, -rayDir)), spheres[i].material.specular_exponent) * attenuation;
			}
			intensity *= 2.0f; // Увеличиваем общую яркость для усиления визуального эффекта
			hitColor = spheres[i].material.diffuse_color * intensity; // Применяем интенсивность света к диффузному цвету материала сферы
		}
	}
	// Проверяем пересечение с источниками света
	for (int j = 0; j < numLights; j++) {
		float t_light;
		if (intersectSphere(rayOrigin, rayDir, lights[j], 0.05f, t_light) && t_light < t_min) {
			t_min = t_light;
			hitColor = make_float3(1.0f, 1.0f, 0.0f); // Желтый цвет для источника света
		}
	}
	output[y * width + x] = hitColor;
}

int main() {
	const int width = 2048;  
	const int height = 2048; 
	const int numSpheres = 5; 
	const int numLights = 1;  
	// Выделение памяти на устройстве (GPU) для хранения результата рендеринга (выходное изображение)
	float3* d_output;
	cudaMalloc(&d_output, width * height * sizeof(float3));
	// Инициализация сфер
	Sphere* d_spheres;
	cudaMalloc(&d_spheres, numSpheres * sizeof(Sphere)); 
	int threadsPerBlock = 256; // Количество потоков на блок в ядре CUDA
	int blocks = (numSpheres + threadsPerBlock - 1) / threadsPerBlock; // Вычисляем количество блоков для запуска ядра
	// Генерация случайных сфер на устройстве
	generateRandomSpheres << <blocks, threadsPerBlock >> > (d_spheres, numSpheres, 1234);
	cudaDeviceSynchronize(); // Синхронизация устройства и хоста для завершения генерации сфер
	// Инициализация источников света
	float3* d_lights; // Указатель на массив позиций источников света на устройстве
	cudaMalloc(&d_lights, numLights * sizeof(float3)); // Выделение памяти на устройстве для источников света
	// Генерация случайных позиций источников света
	generateRandomLightPositions << <blocks, threadsPerBlock >> > (d_lights, numLights, 5678);
	cudaDeviceSynchronize(); // Синхронизация устройства и хоста для завершения генерации света
	// Создание событий для таймеров
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Параметры для запуска ядра CUDA для рендеринга изображения
	dim3 blockSize(16, 16); // Размер блока 16x16 потоков
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); // Размер сетки для покрытия всех пикселей
	// Запись начала выполнения рендеринга
	cudaEventRecord(start);
	// Запуск ядра рендеринга для вычисления цветов изображения
	render << <gridSize, blockSize >> > (d_output, width, height, d_spheres, numSpheres, d_lights, numLights);
	cudaDeviceSynchronize(); // Синхронизация устройства и хоста для завершения рендеринга
	// Запись окончания выполнения рендеринга
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	// Вычисление времени выполнения
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Rendering time: %f ms\n", milliseconds);
	// Выделение памяти на хосте (CPU) для хранения результата рендеринга
	float3* h_output = new float3[width * height];
	// Копирование данных (изображения) с устройства на хост
	cudaMemcpy(h_output, d_output, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
	BMP image;
	image.SetSize(width, height); 
	image.SetBitDepth(24);
	// Преобразование данных с устройства в формат, пригодный для BMP
	for (int y = 0; y < height; y++) { 
		for (int x = 0; x < width; x++) {
			float3 color = h_output[y * width + x]; 
			image(x, y)->Red = static_cast<int>(fminf(color.x * 255, 255));
			image(x, y)->Green = static_cast<int>(fminf(color.y * 255, 255));
			image(x, y)->Blue = static_cast<int>(fminf(color.z * 255, 255));
		}
	}
	image.WriteToFile("output.bmp");
	delete[] h_output; 
	cudaFree(d_output); 
	cudaFree(d_spheres); 
	cudaFree(d_lights); 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}