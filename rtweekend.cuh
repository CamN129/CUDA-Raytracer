#ifndef RTWEEKEND_CUH
#define RTWEEKEND_CUH

#include <math.h>
#include <limits>
#include <memory>


// Usings

using std::shared_ptr;
using std::make_shared;

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

// Utility Functions

__device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}
__device__ inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Common Headers

#include "ray.cuh"
#include "vec3.cuh"

#endif
