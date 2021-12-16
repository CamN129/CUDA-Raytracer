#ifndef VEC3_CUH
#define VEC3_CUH

#include <math.h>
#include <stdlib.h>
#include <iostream>


class vec3 {
    public:
    	// we want these functions to run on the host and device (CPU and GPU)
    	// inline functions since they're simple
        __host__ __device__ inline vec3() : e{0,0,0} {}
        __host__ __device__ inline vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

        __host__ __device__ inline float x() const { return e[0]; }
        __host__ __device__ inline float y() const { return e[1]; }
        __host__ __device__ inline float z() const { return e[2]; }

        __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        __host__ __device__ inline float operator[](int i) const { return e[i]; }
        __host__ __device__ inline float& operator[](int i) { return e[i]; }

        __host__ __device__ inline vec3& operator+=(const vec3 &v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        __host__ __device__ inline vec3& operator*=(const float t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }
        __host__ __device__ inline vec3& operator*=(const vec3 &v2){
		e[0]  *= v2.e[0];
		e[1]  *= v2.e[1];
		e[2]  *= v2.e[2];
		return *this;
        
        }

        __host__ __device__ inline vec3& operator/=(const float t) {
            return *this *= 1/t;
        }

        __host__ __device__ inline float length() const {
            return sqrt(length_squared());
        }

        __host__ __device__ inline float length_squared() const {
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }
        __host__ __device__ bool near_zero() {
		// Return true if the vector is close to zero in all dimensions.
		const auto s = 1e-8;
		return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
	}


    public:
        float e[3];
};

inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return (1/t) * v;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}
// Type aliases for vec3
using point3 = vec3;   // 3D point
using color = vec3;    // RGB color

#endif
