cmake_minimum_required(VERSION 3.15)
project(fdtd2d LANGUAGES CXX)
option(BUILD_CUDA "Build with CUDA support" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
add_subdirectory(third-party/cnpy)
add_subdirectory(third-party/fmt)
find_package(ZLIB REQUIRED)              # cnpy -> ZIP
# find_package(CUDA REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})


# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(fdtd2d fdtd2d_cpu.cpp)
add_executable(fdtd2d_mt fdtd2d_cpu_mt.cpp)
if (BUILD_CUDA)
    add_executable(fdtd2d_cuda fdtd2d_cuda.cu)
endif()

# --------------------------------------------------------------------------------
# Helper library
# --------------------------------------------------------------------------------
add_library(common STATIC
    src/common/config.cpp
    src/common/sim.cpp
)

# --------------------------------------------------------------------------------
# Include directories
# --------------------------------------------------------------------------------
target_include_directories(common PUBLIC 
    inc
    third-party/json/single_include
    third-party/cnpy
)

# --------------------------------------------------------------------------------
# Link libraries
# --------------------------------------------------------------------------------
target_link_libraries(common PUBLIC fmt::fmt cnpy ZLIB::ZLIB)

target_link_libraries(fdtd2d PRIVATE common)
target_link_libraries(fdtd2d_mt PRIVATE common)
if (BUILD_CUDA)
    target_link_libraries(fdtd2d_cuda PRIVATE common)
endif()

# --------------------------------------------------------------------------------
# Targets properties
# --------------------------------------------------------------------------------
if (BUILD_CUDA)
    set_target_properties(fdtd2d_cuda PROPERTIES
        CUDA_ARCHITECTURES "100"
    )
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false")
    # target_compile_options(fdtd2d_cuda PRIVATE
    #     $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    # )
endif()
