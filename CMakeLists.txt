cmake_minimum_required(VERSION 3.10)

project(OpenCVLibTorchTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set OpenCV directory
set(OpenCV_DIR "C:/Program Files/opencv/build/x64/vc15")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set LibTorch directory
set(CMAKE_PREFIX_PATH "C:/Program Files/libtorch")
find_package(Torch REQUIRED)

# Add the executable
add_executable(${PROJECT_NAME} main.cpp)

# Link the libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} "${TORCH_LIBRARIES}")

# Ensure LibTorch headers are included
target_include_directories(${PROJECT_NAME} PRIVATE "${TORCH_INCLUDE_DIRS}")
