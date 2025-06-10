cmake_minimum_required(VERSION 3.15)
project(fdtd2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
add_subdirectory(third-party/cnpy)
add_subdirectory(third-party/fmt)
find_package(ZLIB REQUIRED)              # cnpy -> ZIP

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(fdtd2d fdtd2d_cpu.cpp)

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
