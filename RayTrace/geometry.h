#ifndef GEOMETRY_H
#define GEOMETRY_H
#include <cmath>
#include <vector_types.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cmath>
#include <iostream>
#include <vector>

// ��������� ��� �������� �����
struct Material {
    float3 diffuse_color;
    float specular_exponent; // ���������� ������
    float3 albedo; // ������������ ���������
};

struct Sphere {
    float3 center;
    float radius;
    float3 color;
    Material material; // �������� �����
};
// ������� ��� ��������� ��������
__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// ������� ��� �������� ��������
__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// ������� ��� ��������� ������� �� ������
__device__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

// ������� ��� ���������� ���������� ������������
__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// ������� ��� ������������ �������
__device__ float3 normalize(const float3& v) {
    float length = sqrt(dot(v, v));
    return make_float3(v.x / length, v.y / length, v.z / length);
}

// �������� ����������� ��� float3
__device__ bool operator!=(const float3& a, const float3& b) {
    return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}

// �������� ����������� ��� Sphere
__device__ bool operator!=(const Sphere& a, const Sphere& b) {
    return (a.center != b.center) || (a.radius != b.radius) ||
        (a.color.x != b.color.x) || (a.color.y != b.color.y) || (a.color.z != b.color.z);
}

// ������� ��� ���������� ����� �������
__device__ float length(const float3& v) {
    return sqrt(dot(v, v));
}

// ������� ��� ��������� ������� �� ������ float3
__device__ float3 operator*(float scalar, const float3& vec) {
    return make_float3(scalar * vec.x, scalar * vec.y, scalar * vec.z);
}

// ������� ��� ��������� �������
__device__ float3 reflect(const float3& I, const float3& N) {
    return I - N * 2.0f * dot(N, I);
}

// ����������� ��������� += ��� float3
__device__ float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

// ������� ��� �������� ������
__device__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}
__device__ float3 operator*=(float3& a, const float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}


#endif // GEOMETRY_H
