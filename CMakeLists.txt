cmake_minimum_required(VERSION 3.21)

# Must precede project(): CMP0091 is what makes CMAKE_MSVC_RUNTIME_LIBRARY
# authoritative, and nvcc's host pass otherwise defaults to the static CRT,
# which collides with the C++ side at link time (LNK4098).
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)

project(coresg
        VERSION 0.1.0
        DESCRIPTION "Incremental, multi-GPU CORE-SG construction"
        LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(CORESG_ENABLE_CUDA "Build the CUDA (single/multi-GPU) backend" ON)
option(CORESG_ENABLE_AVX2  "Enable AVX2/FMA in the CPU distance kernels" ON)
option(CORESG_BUILD_TESTS  "Build the unit tests"                       ON)
option(CORESG_BUILD_APPS   "Build the CLI and benchmark drivers"        ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# ---------------------------------------------------------------------------
# CUDA detection (optional -- the whole project builds and runs without it)
# ---------------------------------------------------------------------------
set(CORESG_WITH_CUDA 0)
if(CORESG_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Must be set BEFORE enable_language(CUDA): that call installs a default
    # (sm_52 on this toolkit), after which "if(NOT DEFINED ...)" is useless.
    # sm_70 is a hard floor -- the local-join kernel holds per-node spin locks
    # inside a warp, which only makes progress with independent thread
    # scheduling. Override for a cluster, e.g.
    #   -DCMAKE_CUDA_ARCHITECTURES="70;80;86;89;90"
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
      else()
        set(CMAKE_CUDA_ARCHITECTURES "70;80;86;89" CACHE STRING "CUDA architectures")
      endif()
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    set(CORESG_WITH_CUDA 1)

    foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
      string(REGEX MATCH "^[0-9]+" _archnum "${_arch}")
      if(_archnum AND _archnum LESS 70)
        message(FATAL_ERROR
          "coresg requires CUDA architecture 70 or newer (got ${_arch}). "
          "Re-run with -DCMAKE_CUDA_ARCHITECTURES=\"75;86;89\" or "
          "-DCORESG_ENABLE_CUDA=OFF.")
      endif()
    endforeach()
    message(STATUS "coresg: CUDA backend ENABLED (arch=${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(STATUS "coresg: no CUDA compiler found -- building CPU only")
  endif()
else()
  message(STATUS "coresg: CUDA backend disabled by option")
endif()

# ---------------------------------------------------------------------------
# Warning / optimisation flags
# ---------------------------------------------------------------------------
add_library(coresg_flags INTERFACE)
if(MSVC)
  target_compile_options(coresg_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /permissive- /Zc:__cplusplus /bigobj /EHsc>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:/O2 /fp:fast>)
  target_compile_definitions(coresg_flags INTERFACE NOMINMAX _CRT_SECURE_NO_WARNINGS
                                                    _USE_MATH_DEFINES)
  if(CORESG_ENABLE_AVX2)
    target_compile_options(coresg_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  endif()
else()
  target_compile_options(coresg_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:-O3 -ffast-math>)
  if(CORESG_ENABLE_AVX2)
    target_compile_options(coresg_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
  endif()
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Core library (pure C++, no CUDA headers)
# ---------------------------------------------------------------------------
add_library(coresg_core STATIC
  src/knn_graph.cpp
  src/nndescent.cpp
  src/incremental_knn.cpp
  src/hubness.cpp
  src/intrinsic_dim.cpp
  src/resource.cpp
  src/connectivity.cpp
  src/mst.cpp
  src/coresg.cpp
  src/backend.cpp
  src/cpu_backend.cpp
  src/eval.cpp
  src/io.cpp)

target_include_directories(coresg_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(coresg_core PUBLIC coresg_flags Threads::Threads)
target_compile_definitions(coresg_core PUBLIC CORESG_WITH_CUDA=${CORESG_WITH_CUDA})

# ---------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------
if(CORESG_WITH_CUDA)
  add_library(coresg_cuda STATIC
    cuda/cuda_runtime_api.cu
    cuda/nndescent_kernels.cu
    cuda/mrd_kernels.cu)
  target_include_directories(coresg_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(coresg_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    # The static cudart is built against the static CRT; take the shared one so
    # the whole link agrees on a single runtime.
    CUDA_RUNTIME_LIBRARY Shared)
  target_compile_options(coresg_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3 --use_fast_math>)
  if(MSVC)
    # nvcc's host pass defaults to the static CRT regardless of
    # CMAKE_MSVC_RUNTIME_LIBRARY, so say it explicitly or every executable that
    # pulls in this archive warns LNK4098.
    target_compile_options(coresg_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=/MDd>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-Xcompiler=/MD>)
  endif()

  # The C++ side of the CUDA backend (needs coresg headers + cuda_api.hpp only).
  target_sources(coresg_core PRIVATE src/cuda_backend.cpp)
  target_link_libraries(coresg_core PUBLIC coresg_cuda)
endif()

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
if(CORESG_BUILD_APPS)
  add_executable(coresg_cli apps/coresg_cli.cpp)
  target_link_libraries(coresg_cli PRIVATE coresg_core)

  add_executable(coresg_bench apps/bench.cpp)
  target_link_libraries(coresg_bench PRIVATE coresg_core)

  add_executable(coresg_exp apps/experiments.cpp)
  target_link_libraries(coresg_exp PRIVATE coresg_core)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CORESG_BUILD_TESTS)
  enable_testing()
  foreach(t knn_graph nndescent hubness mst connectivity coresg incremental)
    add_executable(test_${t} tests/test_${t}.cpp)
    target_link_libraries(test_${t} PRIVATE coresg_core)
    add_test(NAME ${t} COMMAND test_${t})
  endforeach()
endif()
