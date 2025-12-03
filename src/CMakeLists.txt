cmake_minimum_required(VERSION 3.18)
project(LuminisApp LANGUAGES CXX CUDA)

# Estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Estándar CUDA → 17 también
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Habilitar features modernos en NVCC
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Opciones de optimización (conservador por defecto)
option(ENABLE_OPENMP_REDUCTION_OPT "Enable OpenMP array reduction optimization" OFF)
option(ENABLE_CUDA_MEMORY_POOL_OPT "Enable CUDA memory pool optimization" OFF)
option(ENABLE_WORK_STEALING_OPT "Enable MPI work stealing optimization" OFF)
option(ENABLE_OVERLAP_OPT "Enable MPI-CUDA overlap optimization" OFF)
option(ENABLE_ASYNC_IO_OPT "Enable async I/O pipeline optimization" OFF)

# Añadir definición de compilador si las opciones están habilitadas
if(ENABLE_OPENMP_REDUCTION_OPT)
    add_compile_definitions(ENABLE_OPENMP_REDUCTION_OPT)
    message(STATUS "OpenMP array reduction optimization ENABLED")
else()
    message(STATUS "OpenMP array reduction optimization DISABLED (using baseline)")
endif()

if(ENABLE_CUDA_MEMORY_POOL_OPT)
    add_compile_definitions(ENABLE_CUDA_MEMORY_POOL_OPT)
    message(STATUS "CUDA memory pool optimization ENABLED")
else()
    message(STATUS "CUDA memory pool optimization DISABLED (using baseline)")
endif()

if(ENABLE_WORK_STEALING_OPT)
    add_compile_definitions(ENABLE_WORK_STEALING_OPT)
    message(STATUS "MPI work stealing optimization ENABLED")
else()
    message(STATUS "MPI work stealing optimization DISABLED (using baseline)")
endif()

if(ENABLE_OVERLAP_OPT)
    add_compile_definitions(ENABLE_OVERLAP_OPT)
    message(STATUS "MPI-CUDA overlap optimization ENABLED")
else()
    message(STATUS "MPI-CUDA overlap optimization DISABLED (using baseline)")
endif()

if(ENABLE_ASYNC_IO_OPT)
    add_compile_definitions(ENABLE_ASYNC_IO_OPT)
    message(STATUS "Async I/O pipeline optimization ENABLED")
else()
    message(STATUS "Async I/O pipeline optimization DISABLED (using baseline)")
endif()

# Archivos fuente
set(CPP_SOURCES
    comunicacion_sync.cpp
    control_global.cpp
    gestor_distribucion.cpp
    main.cpp
    postprocesamiento.cpp
    preprocesamiento.cpp
    modo_serial.cpp
    modo_mpi_openmp_cuda.cpp
    filesystem_utils.cpp
    placeholder.cpp 
)

set(CUDA_SOURCE
    modo_openmp_cuda.cu
    procesamiento_gpu.cu
)

# Crear ejecutable
add_executable(${PROJECT_NAME}
    ${CPP_SOURCES}
    ${CUDA_SOURCE}
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 🔍 MPI
find_package(MPI)
if(MPI_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_LIBRARIES})
endif()

# 🔍 OpenMP
find_package(OpenMP REQUIRED) # Pon REQUIRED para asegurar que lo encuentra
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    
    # --- AGREGA ESTE BLOQUE ---
    # Esto fuerza a que la bandera -fopenmp llegue al compilador
    target_compile_options(${PROJECT_NAME} PRIVATE
        # Para archivos C++ (.cpp)
        $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
        # Para archivos CUDA (.cu) -> Le decimos a nvcc que pase la bandera al host compiler
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    )
    # --------------------------
endif()

# 🔍 OpenCV
find_package(OpenCV)
if(OpenCV_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
endif()

# Evitar errores con std::filesystem
target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)

add_subdirectory(test)
