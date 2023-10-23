Going through [Ray Tracing in One Weekend](https://raytracing.github.io/) series.

## Todos and notes
  - port to CUDA
  - add proper timing measure/image format
  - try profiling the code with ncu
    - don't seem to have the right permissions in IT classroom pc's to use ncu
  - make use of shared/constant/texture/surface memory for hittables/materials?
  - making a default destructor in `hittable.h` seems to lead to a CUDA runtime error, not making makes clangd complain. Is there a compromise?
    - possibly related: `ptxas warning : Stack size for entry function ... cannot be statically determined`

## Useful Links
  - [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html)
  - [Macalester GPU Programming](http://selkie.macalester.edu/csinparallel/modules/GPUProgramming/build/html/index.html)

## Old

- I had (optionally) implemented vectors in SIMD, we'll see if there'll be more opportunities to use it.
  - Controlled by `USE_SIMD` on top of `main.cc`.
- PRNG was provided by [PCG](http://www.pcg-random.org/), because I read somewhere it is good and fast.
  - render time of Book 1 Image 22 (avg, n=10) was shorter when I used `std::default_random_engine` over `pcg32`, so that may not be true (at least for this setting).

