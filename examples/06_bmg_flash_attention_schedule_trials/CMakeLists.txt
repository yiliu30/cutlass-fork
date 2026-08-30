# Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set(CUTLASS_APPLICATIONS_DIR ${CUTLASS_DIR}/applications)

foreach(TRY RANGE 22 36)
  set(TARGET 06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try${TRY})
  cutlass_example_add_executable(${TARGET} 06_xe_fmha_fwd.cpp)
  target_compile_definitions(
    ${TARGET}
    PRIVATE
      HEAD_DIM=128
      PREFILL
      SHOW_DIFF=1
      INPUT_TYPE=bfloat16_t
      IS_BFLOAT16
      QK_SCHEDULE=${TRY}
  )
endforeach()

set(TARGET 06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try38)
cutlass_example_add_executable(${TARGET} 06_xe_fmha_fwd.cpp)
target_compile_definitions(
  ${TARGET}
  PRIVATE
    HEAD_DIM=128
    PREFILL
    SHOW_DIFF=1
    INPUT_TYPE=bfloat16_t
    IS_BFLOAT16
    QK_SCHEDULE=38
)

set(TARGET 06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try39)
cutlass_example_add_executable(${TARGET} 06_xe_fmha_fwd.cpp)
target_compile_definitions(
  ${TARGET}
  PRIVATE
    HEAD_DIM=128
    PREFILL
    SHOW_DIFF=1
    INPUT_TYPE=bfloat16_t
    IS_BFLOAT16
    QK_SCHEDULE=39
)

set(TARGET 06_xe_fmha_fwd_prefill_bfloat16_t_hdim128_try37)
cutlass_example_add_executable(${TARGET} 06_xe_fmha_fwd.cpp)
target_compile_definitions(
  ${TARGET}
  PRIVATE
    HEAD_DIM=128
    PREFILL
    SHOW_DIFF=1
    INPUT_TYPE=bfloat16_t
    IS_BFLOAT16
    QK_SCHEDULE=37
    PV_WIDTH_32
)
