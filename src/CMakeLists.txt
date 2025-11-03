cmake_minimum_required(VERSION 3.18)
project(Proyecto_Hibrido LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${MPI_INCLUDE_PATH} ${OpenCV_INCLUDE_DIRS})

file(GLOB SRC_FILES src/*.cpp src/*.cu)

add_executable(proyecto_hibrido ${SRC_FILES})

# linkeo
target_link_libraries(proyecto_hibrido
    PRIVATE
    ${MPI_LIBRARIES}
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
    cuda
    cudart
)

# si MPI::MPI_CXX está disponible, preferirla
if(TARGET MPI::MPI_CXX)
    target_link_libraries(proyecto_hibrido PRIVATE MPI::MPI_CXX)
endif()
