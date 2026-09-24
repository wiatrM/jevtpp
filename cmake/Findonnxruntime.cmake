find_path(onnxruntime_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
  PATH_SUFFIXES include include/onnxruntime)
find_library(onnxruntime_LIBRARY NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(onnxruntime
  REQUIRED_VARS onnxruntime_LIBRARY onnxruntime_INCLUDE_DIR)

if(onnxruntime_FOUND AND NOT TARGET onnxruntime::onnxruntime)
  add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${onnxruntime_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIR}")
endif()
