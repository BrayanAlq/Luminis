cmake_minimum_required(VERSION 3.18)
project(LuminisApp LANGUAGES CXX CUDA)

# Estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Incluir directorio actual
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 🔍 MPI
find_package(MPI)
if(MPI_FOUND)
    message(STATUS "MPI encontrado: ${MPI_CXX_COMPILER}")
    target_include_directories(${PROJECT_NAME} PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_LIBRARIES})
else()
    message(WARNING "MPI no encontrado. Compilación sin soporte MPI.")
endif()

# 🔍 OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP habilitado")
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP no disponible. Compilación sin paralelismo OpenMP.")
endif()

# 🔍 OpenCV
find_package(OpenCV)
if(OpenCV_FOUND)
    message(STATUS "OpenCV encontrado: ${OpenCV_VERSION}")
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
else()
    message(WARNING "OpenCV no encontrado. Algunas funciones pueden no estar disponibles.")
endif()

# 🔍 CUDA
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)

target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
