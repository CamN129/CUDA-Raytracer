#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "hittable_list.cuh"
#include "sphere.cuh"
#include "rtweekend.cuh"
#include <curand_kernel.h>
#include "camera.cuh"
#include "material.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// We can solve (point - center) dot (point-center) = r^2 where p=a+tb, in terms of t using quadratic formula
__device__ float hit_sphere(const point3& center, float radius, const ray& r) {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;
    auto discriminant = half_b*half_b - a*c;
    if (discriminant < 0) {
        return -1.0;
    } else {
        return (-half_b - sqrt(discriminant) ) / a;
    }
}

__device__ vec3 ray_color(const ray& r, hittable **world, curandState *local_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    // to prevent stack overflow
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        // if we hit an object we want the ray to bounce off which is why we recursively call ray_color
        if ((*world)->hit(cur_ray, 0.001f, infinity, rec)) {
            ray scattered;
            vec3 attenuation;
            // bounces off according to materials
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
        	// otherwise we hit background
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}


__global__
void render(vec3 *y,int w,int h, hittable **world, int ns, curandState *rand_state, camera **cam)
{
	// height, width, and output matrix as inputs
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if ((i < w) && (j < h)){
		// j*w+icomes from our "flattened matrix" structure
		int index = j*w+i;
		curandState local_rand_state = rand_state[index];
		color pixel_color(0,0,0);
		
		for(int s=0; s < ns; s++) {
		// taking multiple samples per pixel smoothens the image, anti- aliasing
			float u = float(i + curand_uniform(&local_rand_state)) / float(w);
			float v = float(j + curand_uniform(&local_rand_state)) / float(h);
			ray r = (*cam)->get_ray(u,v,&local_rand_state);
			pixel_color += ray_color(r, world, &local_rand_state);
		}
		rand_state[index] = local_rand_state;
		
		// we have to gamma correct our value
		float r = sqrt(pixel_color.x()/float(ns));
		float g = sqrt(pixel_color.y()/float(ns));
		float b = sqrt(pixel_color.z()/float(ns));
		y[index] = color(r,g,b);
	}
}

#define RND (curand_uniform(&local_rand_state))
__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
    	curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        
        *rand_state = local_rand_state;
        *d_world  = new hittable_list(d_list, 22*22+1+3);
               
        // camera properties
	point3 lookfrom(13,2,3);
	point3 lookat(0,0,0);
	vec3 vup(0,1,0);
	auto dist_to_focus = 10.0;
    	auto aperture = 0.1;
	
        *d_camera = new camera(lookfrom, lookat, vup, 20 ,16.0/9.0,aperture,dist_to_focus);
    }
}


__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera) {
	for(int i=0; i <22*22+1+3; i++) {
	    delete ((sphere *)d_list[i])->mat_ptr;
	    delete d_list[i];
	  }
	delete *d_world;
	delete *d_camera;
}

// initialize single curand state for world generation
__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

// we have to set up the curand states for the random part of the antialiasing
__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j*max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}


int main() {

	// Image
	const auto aspect_ratio = 3.0 / 2.0;
   	const int image_width = 1200;
	const int image_height = static_cast<int>(image_width / aspect_ratio);
	const int samples_per_pixel = 100;


	// World
	hittable **d_list;
	int num_hitables = 22*22+1+3;
	checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hittable *)));
	hittable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
	
	// Camera
	camera **cam;
	checkCudaErrors(cudaMalloc((void **)&cam, sizeof(camera *)));
	
	// random state for world generation
	curandState *d_rand_state_small;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state_small, sizeof(curandState)));
	rand_init<<<1,1>>>(d_rand_state_small);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	create_world<<<1,1>>>(d_list,d_world,cam, d_rand_state_small);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

    
	// Render
	// get size of our frame buffer
	int num_pixels = image_width*image_height;
	size_t fb_size = 3*num_pixels*sizeof(vec3);

	//make frame buffer
	vec3 *fb;
	
	// cudaMallocManaged gives memory accessible by host as well
	checkCudaErrors(cudaMallocManaged((void **) &fb, fb_size));
	
	//thread block size length
	int tb = 8;
	dim3 threads(tb,tb);
	dim3 blocks(image_width/tb+1,image_height/tb+1);
			
	//create render kernel
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

	render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	render<<<blocks, threads>>>(fb, image_width, image_height, d_world, samples_per_pixel, d_rand_state, cam);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	// Print out
	std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
	for (int j = image_height-1; j >= 0; --j) {
		//std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
		for (int i = 0; i < image_width; ++i) {
			// get index we want to print
			size_t pixel_index = j*image_width + i;
			float r = fb[pixel_index].x();
			float g = fb[pixel_index].y();
			float b = fb[pixel_index].z();
			
			// translate it to normal RGB values
			int ir = static_cast<int>(255.999 * r);
			int ig = static_cast<int>(255.999 * g);
			int ib = static_cast<int>(255.999 * b);

			std::cout << ir << ' ' << ig << ' ' << ib << '\n';
		}
	}
	checkCudaErrors(cudaDeviceSynchronize());
	//std::cerr << "\nDone.\n";
	// free everything
	free_world<<<1,1>>>(d_list,d_world,cam);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(d_rand_state_small));
	checkCudaErrors(cudaFree(cam));
	checkCudaErrors(cudaFree(fb));
}
