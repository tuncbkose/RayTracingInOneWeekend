#include <iostream>

#include "ray.h"

#include "utils/cuda_utils.h"
#include "utils/vec3.h"

static inline int divup(int a, int b){
    // Divide and round up, to calculate number of blocks
    return (a + b - 1)/b;
}

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r){
    /*
     *  Returns whether ray r hits a sphere by solving the quadratic equation
     *  (r(t)-center) \cdot (r(t)-center) = radius^2 
     */
    vec3 oc = r.origin() - center;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.f * dot(r.direction(), oc);
    auto c = dot(oc, oc) - radius*radius;
    auto discriminant = b*b - 4*a*c;
    return (discriminant >= 0);
}

__device__ vec3 ray_color(const ray& r){
    /*
     * Calculate the color of given ray
     * 'f's enforce single precision arithmetic for GPU performance
     */
    if (hit_sphere(vec3(0,0,-1), 0.5, r)) return vec3(1, 0, 0);
    
    vec3 unit_dir = unit_vector(r.direction());
    auto a = 0.5f*(unit_dir.y() + 1.f);
    return (1.f-a)*vec3(1., 1., 1.) + a*vec3(0.5, 0.7, 1.);
}

__global__ void render(vec3* fb, int max_x, int max_y, vec3 top_left_loc, 
                        vec3 delta_horizontal, vec3 delta_vertical, vec3 origin){
    /* 
     * - top_left_loc: location of the top-left pixel in the image
     * - delta_horizontal: horizontal difference between pixels (going left)
     * - delta_vertical : vertical difference between pixels (going down)
     * - origin: location of camera
     */
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // if out of bounds, do nothing
    if ((ix >= max_x) || (iy >= max_y)) return;

    auto pixel_center = top_left_loc + (ix*delta_horizontal) + (iy*delta_vertical);
    ray r(origin, pixel_center-origin);
    
    fb[ix + iy*max_x] = ray_color(r);
}

int main(){

    // Image

    auto aspect_ratio = 16. / 9.;
    int image_width = 400;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height >= 1) ? image_height : 1;

    // Camera
    auto viewport_height = 2.;
    auto viewport_width = viewport_height * (static_cast<double>(image_width)/image_height);
    auto focal_length = 1.;
    auto camera_center = vec3(0, 0, 0);
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);

    // horizontal/vertical dist between pixels in the viewport
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    // Figure out location of top left pixel
    auto viewport_upper_left = camera_center - vec3(0,0, focal_length) - viewport_u/2 - viewport_v/2;
    auto pixel00_loc = viewport_upper_left + 0.5*(pixel_delta_u + pixel_delta_v);
            
    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels*sizeof(vec3);
    
    // Determine blocks and threads for CUDA
    int tx = 8;
    int ty = 8;
    dim3 blocks(divup(image_width, tx), divup(image_height, ty));
    dim3 threads(tx, ty);  // should be multiple of 32

    // Render

    // Allocate frame buffer in GPU
    vec3* fb_gpu;
    checkCudaErrors(cudaMalloc(&fb_gpu, fb_size));

    render<<<blocks, threads>>>(fb_gpu, image_width, image_height, 
        pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center);
    checkCudaErrors(cudaGetLastError());

    // Copy results to host
    vec3* fb_host = new vec3[num_pixels];
    checkCudaErrors(cudaMemcpy(fb_host, fb_gpu, fb_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(fb_gpu));  // we are done with GPU memory
    
    // Output to file
    
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            auto pixel = fb_host[i + j*image_width];
            int ir = static_cast<int>(255.999 * pixel.x());
            int ig = static_cast<int>(255.999 * pixel.y());
            int ib = static_cast<int>(255.999 * pixel.z());

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    delete[] fb_host;
}
