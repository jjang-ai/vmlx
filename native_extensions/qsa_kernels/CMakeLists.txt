cmake_minimum_required(VERSION 3.27)

project(mtplx_qsa_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Keep C++ and Metal minimum OS aligned. The MLX helper does not propagate
# CMAKE_OSX_DEPLOYMENT_TARGET to metal, so use an explicit command below.
# NAX is packaged only for a 26.2+ target and selected by MLX's live gate.
if(CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
  set(CMAKE_OSX_DEPLOYMENT_TARGET
      "15.0"
      CACHE STRING "" FORCE)
endif()

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# Pass BOTH hints from sys.executable at configure time (see setup.py):
# CMake otherwise happily selects a framework Python outside the venv and
# links the extension against the wrong interpreter.
find_package(
  Python 3.11
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# Build receipts baked into the module. The primitive links MLX's private C++
# ABI, so the wrapper must be able to prove which wheel it was built against.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import mlx.core as mx; print(mx.__version__)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MTPLX_QSA_MLX_VERSION)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import nanobind; print(nanobind.__version__)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MTPLX_QSA_NANOBIND_VERSION)

set(MTPLX_QSA_METAL_LIBRARY "mtplx_qsa_kernels")

add_library(mtplx_qsa_kernel_ops)
target_sources(
  mtplx_qsa_kernel_ops
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/qwen4_qsa_sparse_gqa.cpp)
target_include_directories(mtplx_qsa_kernel_ops
                           PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(mtplx_qsa_kernel_ops PUBLIC mlx)

if(MLX_BUILD_METAL)
  set(MTPLX_QSA_METAL_FLAGS -std=metal3.2)
  if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_GREATER_EQUAL "26.2")
    set(MTPLX_QSA_METAL_FLAGS -std=metal4.0 -DMTPLX_QSA_HAS_NAX=1)
    target_compile_definitions(mtplx_qsa_kernel_ops PRIVATE MTPLX_QSA_HAS_NAX=1)
  endif()
  set(MTPLX_QSA_METALLIB "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${MTPLX_QSA_METAL_LIBRARY}.metallib")
  add_custom_command(
    OUTPUT ${MTPLX_QSA_METALLIB}
    COMMAND xcrun -sdk macosx metal
      -I${CMAKE_CURRENT_LIST_DIR}
      "$<LIST:TRANSFORM,${MLX_INCLUDE_DIRS},PREPEND,-I>"
      ${MTPLX_QSA_METAL_FLAGS}
      -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
      -Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wmetal-addr-spaces
      ${CMAKE_CURRENT_LIST_DIR}/qwen4_qsa_sparse_gqa.metal
      -o ${MTPLX_QSA_METALLIB}
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/qwen4_qsa_sparse_gqa.metal
            ${CMAKE_CURRENT_LIST_DIR}/kernels/steel_qwen4_qsa_sparse_gqa.h
    COMMAND_EXPAND_LISTS VERBATIM)
  add_custom_target(mtplx_qsa_kernels_metallib DEPENDS ${MTPLX_QSA_METALLIB})

  add_dependencies(mtplx_qsa_kernel_ops mtplx_qsa_kernels_metallib)
endif()

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE mtplx_qsa_kernel_ops)
target_compile_definitions(
  _ext
  PRIVATE MTPLX_QSA_BUILD_MLX_VERSION="${MTPLX_QSA_MLX_VERSION}"
          MTPLX_QSA_BUILD_NANOBIND_VERSION="${MTPLX_QSA_NANOBIND_VERSION}"
          MTPLX_QSA_METAL_LIBRARY="${MTPLX_QSA_METAL_LIBRARY}")

if(BUILD_SHARED_LIBS)
  # @loader_path resolves the sibling kernel_ops dylib. The second entry
  # resolves the mlx wheel's libmlx.dylib from the installed layout
  # (site-packages/mtplx_qsa_kernels -> site-packages -> mlx/lib); without it
  # the only libmlx rpath is the link-time absolute path into the build env,
  # which is dead after install.
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path
                      -Wl,-rpath,@loader_path/../mlx/lib)
endif()
