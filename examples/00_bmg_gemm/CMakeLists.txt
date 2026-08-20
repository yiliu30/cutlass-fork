# Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if (CUTLASS_TEST_FOR_CRI)
  set(TEST_CRI_ARGS "--m=${CRI_GEMM_DEFAULT_SIZE} --n=${CRI_GEMM_DEFAULT_SIZE} --k=${CRI_GEMM_DEFAULT_SIZE} --iterations=${CRI_DEFAULT_ITERATIONS} --verify=${CRI_DEFAULT_VERIFY}")

  set(GEMM_ARGS TEST_CRI_ARGS)
  set(GEMM_PADDED_ARGS TEST_CRI_ARGS)
  set(GEMM_QUEUE_ARGS TEST_CRI_ARGS)
else()
  set(TEST_BATCHES --l=2)
  set(TEST_LARGE "--l=513 --m=8 --n=16384 --k=512") # B matrix capacity > uint32_max
  set(TEST_SMALL_SHAPE --m=4 --n=8 --k=8 --l=2)
  set(TEST_SMALL_SHAPE_PADDABLE --m=1 --n=1 --k=2 --l=2)

  set(GEMM_ARGS TEST_BATCHES TEST_LARGE TEST_SMALL_SHAPE)
  set(GEMM_PADDED_ARGS TEST_BATCHES TEST_SMALL_SHAPE_PADDABLE)
  set(GEMM_QUEUE_ARGS TEST_BATCHES)
endif()

cutlass_example_add_executable(
  00_bmg_gemm
  00_bmg_gemm.cpp
  TEST_COMMAND_OPTIONS
  ${GEMM_ARGS}
)

set(GEMM_INT_ARGS "--m=256 --n=256 --k=512 --l=1 --iterations=1 --verify=1")

cutlass_example_add_executable(
  00_bmg_gemm_int8
  00_bmg_gemm.cpp
  TEST_COMMAND_OPTIONS
  ${GEMM_INT_ARGS}
)
target_compile_definitions(00_bmg_gemm_int8 PRIVATE BMG_GEMM_INT8)

cutlass_example_add_executable(
  00_bmg_gemm_int4
  00_bmg_gemm.cpp
  TEST_COMMAND_OPTIONS
  ${GEMM_INT_ARGS}
)
target_compile_definitions(00_bmg_gemm_int4 PRIVATE
  BMG_GEMM_INT4
  BMG_GEMM_TILE_M=256
  BMG_GEMM_TILE_N=256
  BMG_GEMM_TILE_K=128
  BMG_GEMM_LAYOUT_B_COLUMN)

if (CUTLASS_ENABLE_SYCL)
  target_link_options(00_bmg_gemm_int8 PRIVATE -Xs "-options \"-igc_opts 'allowDecompose2DBlockFuncs=0'\"")
  target_link_options(00_bmg_gemm_int4 PRIVATE -Xs "-options \"-igc_opts 'allowDecompose2DBlockFuncs=0'\"")
endif()

foreach(config IN ITEMS
    "s4_256x256x64;BMG_GEMM_INT4;256;256;64;8x4;row"
    "s4_256x256x128;BMG_GEMM_INT4;256;256;128;8x4;row"
    "s4_128x256x128;BMG_GEMM_INT4;128;256;128;8x4;row"
    "s4_256x128x128_sg4x8;BMG_GEMM_INT4;256;128;128;4x8;row"
    "s4_256x256x128_colb;BMG_GEMM_INT4;256;256;128;8x4;col"
    "u4_256x256x128;BMG_GEMM_UINT4;256;256;128;8x4;row"
    "u4_256x256x128_colb;BMG_GEMM_UINT4;256;256;128;8x4;col")
  list(GET config 0 suffix)
  list(GET config 1 dtype_macro)
  list(GET config 2 tile_m)
  list(GET config 3 tile_n)
  list(GET config 4 tile_k)
  list(GET config 5 sg_layout)
  list(GET config 6 b_layout)
  set(target 00_bmg_gemm_int4_${suffix})
  cutlass_example_add_executable(
    ${target}
    00_bmg_gemm.cpp
    TEST_COMMAND_OPTIONS
    ${GEMM_INT_ARGS}
  )
  target_compile_definitions(${target} PRIVATE
    ${dtype_macro}
    BMG_GEMM_TILE_M=${tile_m}
    BMG_GEMM_TILE_N=${tile_n}
    BMG_GEMM_TILE_K=${tile_k})
  if (sg_layout STREQUAL "4x8")
    target_compile_definitions(${target} PRIVATE BMG_GEMM_SG_LAYOUT_4X8)
  endif()
  if (b_layout STREQUAL "col")
    target_compile_definitions(${target} PRIVATE BMG_GEMM_LAYOUT_B_COLUMN)
  endif()
  if (CUTLASS_ENABLE_SYCL)
    target_link_options(${target} PRIVATE -Xs "-options \"-igc_opts 'allowDecompose2DBlockFuncs=0'\"")
  endif()
endforeach()

cutlass_example_add_executable(
  00_bmg_gemm_padded
  00_bmg_gemm_padded.cpp
  TEST_COMMAND_OPTIONS
  ${GEMM_PADDED_ARGS}
)

cutlass_example_add_executable(
  00_bmg_gemm_with_sycl_queue
  00_bmg_gemm_with_sycl_queue.cpp
  TEST_COMMAND_OPTIONS
  ${GEMM_QUEUE_ARGS}
)
