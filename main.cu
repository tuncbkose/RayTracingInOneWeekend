#include <iostream>

#include "utils/cuda_utils.h"


static inline int divup(int a, int b){
    // Divide and round up, to calculate number of blocks
    return (a + b - 1)/b;
}

__global__ void render(float* fb, int max_x, int max_y){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // if out of bounds, do nothing
    if ((ix >= max_x) || (iy >= max_y)) return;

    // 3 floats per pixel (RGB)
    int pixel_idx = (ix + iy*max_x)*3;
    fb[pixel_idx + 0] = float(ix) / max_x;
    fb[pixel_idx + 1] = float(iy) / max_y;
    fb[pixel_idx + 2] = 0;
}

int main(){

    // Image

    int image_width = 256;
    int image_height = 256;
    int num_pixels = image_width * image_height;

    // Allocate frame buffer in GPU, 3 floats per pixel (RGB)
    float* fb_gpu;
    size_t fb_num_floats = 3*num_pixels;
    checkCudaErrors(cudaMalloc(&fb_gpu, fb_num_floats*sizeof(float)));

    // Determine blocks and threads for CUDA
    int tx = 8;
    int ty = 8;
    dim3 blocks(divup(image_width, tx), divup(image_height, ty));
    dim3 threads(tx, ty);  // should be multiple of 32

    // Render

    render<<<blocks, threads>>>(fb_gpu, image_width, image_height);
    checkCudaErrors(cudaGetLastError());

    // Copy results to host
    float* fb_host = new float[fb_num_floats];
    checkCudaErrors(cudaMemcpy(fb_host, fb_gpu, fb_num_floats*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(fb_gpu));  // we are done with GPU memory
    
    // Output to file
    
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            int pixel_idx = (i + j*image_width)*3;
            auto r = fb_host[pixel_idx + 0];
            auto g = fb_host[pixel_idx + 1];
            auto b = fb_host[pixel_idx + 2]; 

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    delete[] fb_host;
}
