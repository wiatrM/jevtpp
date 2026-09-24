execute_process(COMMAND "${CMAKE_COMMAND}" -S "${SOURCE_DIR}/tests/compile_fail"
  -B "${BINARY_DIR}/compile_fail/${CASE}" -G "${GENERATOR}"
  "-DCMAKE_CXX_COMPILER=${COMPILER}" "-DJEVT_SOURCE_DIR=${SOURCE_DIR}" "-DCASE=${CASE}"
  RESULT_VARIABLE configure_result OUTPUT_VARIABLE configure_out ERROR_VARIABLE configure_err)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR "Negative test configuration failed: ${configure_out}${configure_err}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${BINARY_DIR}/compile_fail/${CASE}"
  RESULT_VARIABLE build_result OUTPUT_VARIABLE build_out ERROR_VARIABLE build_err)
if(build_result EQUAL 0 OR NOT "${build_out}${build_err}" MATCHES "${EXPECTED}")
  message(FATAL_ERROR "Expected compile error '${EXPECTED}', got: ${build_out}${build_err}")
endif()
