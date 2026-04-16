set(LLAMA_VERSION      0.0.3962)
set(LLAMA_BUILD_COMMIT 1f86f058)
set(LLAMA_BUILD_NUMBER 3962)
set(LLAMA_SHARED_LIB   ON)

set(GGML_BLAS       OFF)
set(GGML_CUDA       OFF)
set(GGML_METAL      OFF)
set(GGML_HIPBLAS    OFF)
set(GGML_ACCELERATE ON)
set(GGML_VULKAN OFF)
set(GGML_VULKAN_CHECK_RESULTS OFF)
set(GGML_VULKAN_DEBUG OFF)
set(GGML_VULKAN_MEMORY_DEBUG OFF)
set(GGML_VULKAN_VALIDATE OFF)
set(GGML_SYCL OFF)
set(GGML_OPENMP ON)


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was llama-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set_and_check(LLAMA_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")
set_and_check(LLAMA_LIB_DIR     "${PACKAGE_PREFIX_DIR}/lib")
set_and_check(LLAMA_BIN_DIR     "${PACKAGE_PREFIX_DIR}/bin")

# Ensure transient dependencies satisfied

find_package(Threads REQUIRED)

if (APPLE AND GGML_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
endif()

if (GGML_BLAS)
    find_package(BLAS REQUIRED)
endif()

if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if (GGML_METAL)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
endif()

if (GGML_VULKAN)
    find_package(Vulkan REQUIRED)
endif()

if (GGML_HIPBLAS)
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(rocblas REQUIRED)
endif()

if (GGML_SYCL)
    find_package(IntelSYCL REQUIRED)
    find_package(MKL REQUIRED)
endif()

if (GGML_OPENMP)
    find_package(OpenMP REQUIRED)
endif()


find_library(ggml_LIBRARY ggml
    REQUIRED
    HINTS ${LLAMA_LIB_DIR})

find_library(llama_LIBRARY llama
    REQUIRED
    HINTS ${LLAMA_LIB_DIR})

set(_llama_link_deps "${ggml_LIBRARY}" "OpenMP::OpenMP_C;OpenMP::OpenMP_CXX;Threads::Threads;m")
set(_llama_transient_defines "GGML_SCHED_MAX_COPIES=4;$<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>;GGML_USE_OPENMP;GGML_USE_LLAMAFILE;_XOPEN_SOURCE=600;_GNU_SOURCE;GGML_SHARED;GGML_BUILD")

add_library(llama UNKNOWN IMPORTED)

set_target_properties(llama
    PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${_llama_link_deps}"
        INTERFACE_COMPILE_DEFINITIONS "${_llama_transient_defines}"
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${llama_LIBRARY}"
        INTERFACE_COMPILE_FEATURES cxx_std_11
        POSITION_INDEPENDENT_CODE ON )

check_required_components(Llama)
