#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "ray.cuh"
#include "rtweekend.cuh"

class material;

struct hit_record {
	point3 p;
	vec3 normal;
	float t;
	bool front_face;
	material *mat_ptr;
	
	// get outward facing normal
	__device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal :-outward_normal;
	}
};

class hittable {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif