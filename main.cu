#include <iostream>

#include "utils/cuda_utils.h"
#include "utils/vec3.h"

static inline int divup(int a, int b){
    // Divide and round up, to calculate number of blocks
    return (a + b - 1)/b;
}

__global__ void render(vec3* fb, int max_x, int max_y){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // if out of bounds, do nothing
    if ((ix >= max_x) || (iy >= max_y)) return;
    
    fb[ix + iy*max_x] = vec3(float(ix)/max_x, float(iy)/max_y, 0);
}

int main(){

    // Image

    int image_width = 256;
    int image_height = 256;
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

    render<<<blocks, threads>>>(fb_gpu, image_width, image_height);
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
