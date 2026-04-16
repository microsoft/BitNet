# Install script for directory: /home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/3rdparty/llama.cpp/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/cvector-generator/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/baby-llama/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/batched-bench/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/batched/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/convert-llama2c-to-ggml/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/embedding/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/eval-callback/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/export-lora/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/gbnf-validator/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/gguf-hash/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/gguf-split/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/gguf/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/gritlm/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/imatrix/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/infill/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/llama-bench/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/llava/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/lookahead/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/lookup/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/main/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/parallel/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/passkey/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/perplexity/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/quantize-stats/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/quantize/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/retrieval/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/server/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/save-load-state/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/simple/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/speculative/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/tokenize/cmake_install.cmake")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/joan/Documents/IA/sharing/3rdparty/BitNet-1.58b/build_npu/3rdparty/llama.cpp/examples/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
