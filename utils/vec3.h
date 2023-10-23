#pragma once

#include <cmath>
#include <iostream>
#include <ostream>

class vec3 {
    public:
        float e[3];
    
        __host__ __device__ vec3() {}
        __host__ __device__ vec3(float e0, float e1, float e2): e{e0, e1, e2} {}


        // TODO: Is all this inlining appropriate?
        __host__ __device__ inline float x() const { return e[0]; }
        __host__ __device__ inline float y() const { return e[1]; }
        __host__ __device__ inline float z() const { return e[2]; }

        __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        __host__ __device__ inline float operator[](int i) const { return e[i]; }
        __host__ __device__ inline float& operator[](int i) { return e[i]; }

        __host__ __device__ inline vec3& operator+=(const vec3& v){
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }
        __host__ __device__ inline vec3& operator-=(const vec3& v){
            e[0] -= v.e[0];
            e[1] -= v.e[1];
            e[2] -= v.e[2];
            return *this;
        }
        __host__ __device__ inline vec3& operator*=(const vec3& v){
            e[0] *= v.e[0];
            e[1] *= v.e[1];
            e[2] *= v.e[2];
            return *this;
        }
        __host__ __device__ inline vec3& operator/=(const vec3& v){
            e[0] /= v.e[0];
            e[1] /= v.e[1];
            e[2] /= v.e[2];
            return *this;
        }
        __host__ __device__ inline vec3& operator*=(float t){
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }
        __host__ __device__ inline vec3& operator/=(float t){
            float k = 1./t;
            e[0] *= k;
            e[1] *= k;
            e[2] *= k;
            return *this;
        }

        __host__ __device__ inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
        __host__ __device__ inline float squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
   
};

inline std::istream& operator>>(std::istream& is, vec3& v) {
    is >> v.e[0] >> v.e[1] >> v.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& v) {
    os << v.e[0] << " " << v.e[1] << " " << v.e[2];
    return os;
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v){
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}
__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v){
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}
__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v){
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}
__host__ __device__ inline vec3 operator/(const vec3& u, const vec3& v){
    return vec3(u.e[0] / v.e[0], u.e[1] / v.e[1], u.e[2] / v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v){
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}
__host__ __device__ inline vec3 operator/(const vec3& v, float t){
    float k = 1./t;
    return vec3(k*v.e[0], k*v.e[1], k*v.e[2]);
}
__host__ __device__ inline vec3 operator*(const vec3& v, float t){
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v){
    return u.e[0]*v.e[0] + u.e[1]*v.e[1] + u.e[2]*v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v){
    return vec3(u.e[1]*v.e[2] - u.e[2]*v.e[1],
                u.e[2]*v.e[0] - u.e[0]*v.e[2],
                u.e[0]*v.e[1] - u.e[1]*v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v){
    return v / v.length();
}

