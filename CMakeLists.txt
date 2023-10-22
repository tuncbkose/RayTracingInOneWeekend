cmake_minimum_required(VERSION 3.26)

project(ray_tracing LANGUAGES CUDA CXX)

add_executable(out main.cu utils/cuda_utils.cu)

set(CXX_FLAGS "-Wall -Wextra -Werror -march=native")
target_compile_options(out PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${CXX_FLAGS}>
  $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>  # I suspect in this state it may give some error due to quotation marks. We'll see.
) 
