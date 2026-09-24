# Optional, pinned native engine. No dependency is fetched for core/ORT builds.
include(FetchContent)
enable_language(C ASM)
if(JEVT_LAYA_NATIVE_CUDA)
  enable_language(CUDA)
endif()
set(JEVT_LAYA_NATIVE_REVISION e1c6e7832189d36903e90c6fe3b8b1fece7f6f17)

function(jevt_add_native_engine)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  # Keep implementation dependencies inside our shared adapter. These are
  # directory-local choices, not overrides of the application's core ABI.
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTING OFF)
  set(LAYA_CUDA ${JEVT_LAYA_NATIVE_CUDA})
  set(LAYA_VULKAN OFF)
  set(LAYA_COREML OFF)
  set(LAYA_PORTABLE OFF)
  set(GGML_METAL OFF)
  set(GGML_NATIVE OFF)
  set(GGML_BACKEND_DL OFF)
  set(GGML_CPU_ALL_VARIANTS OFF)
  set(GGML_BACKEND_DIR "")
  find_package(nlohmann_json 3.11 QUIET CONFIG)
  if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(nlohmann_json
      GIT_REPOSITORY https://github.com/nlohmann/json.git
      GIT_TAG 55f93686c01528224f448c19128836e7df245f72
      GIT_SHALLOW FALSE
      OVERRIDE_FIND_PACKAGE)
    FetchContent_MakeAvailable(nlohmann_json)
  endif()
  if(JEVT_LAYA_NATIVE_SOURCE_DIR)
    set(native_source "${JEVT_LAYA_NATIVE_SOURCE_DIR}")
  else()
    FetchContent_Declare(jevt_native_engine
      GIT_REPOSITORY https://github.com/lkarlslund/laya.cpp.git
      GIT_TAG ${JEVT_LAYA_NATIVE_REVISION}
      GIT_SHALLOW FALSE
      GIT_SUBMODULES_RECURSE TRUE)
    FetchContent_GetProperties(jevt_native_engine)
    if(NOT jevt_native_engine_POPULATED)
      FetchContent_Populate(jevt_native_engine)
    endif()
    set(native_source "${jevt_native_engine_SOURCE_DIR}")
  endif()
  if(NOT EXISTS "${native_source}/third_party/ggml/CMakeLists.txt")
    message(FATAL_ERROR "Native source must include pinned submodules; run git submodule update --init --recursive")
  endif()
  add_subdirectory("${native_source}" "${CMAKE_CURRENT_BINARY_DIR}/native-engine" EXCLUDE_FROM_ALL)
  # Public adapter provenance is tied to the pinned API. Overrides are for
  # development and must be independently validated before making claims.
  set(JEVT_NATIVE_SOURCE "${native_source}" PARENT_SCOPE)
endfunction()

jevt_add_native_engine()
set_target_properties(jevt PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(jevt_laya_native SHARED src/laya_native.cpp)
add_library(jevt::laya_native ALIAS jevt_laya_native)
target_compile_features(jevt_laya_native PUBLIC cxx_std_20)
target_include_directories(jevt_laya_native PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(jevt_laya_native PUBLIC jevt::jevt PRIVATE laya)
set_target_properties(jevt_laya_native PROPERTIES
  VERSION ${PROJECT_VERSION} SOVERSION 0 EXPORT_NAME laya_native WINDOWS_EXPORT_ALL_SYMBOLS ON)
install(TARGETS jevt_laya_native EXPORT JevTTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES "${JEVT_NATIVE_SOURCE}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/jevtpp/licenses" RENAME laya.cpp-LICENSE)
install(FILES "${JEVT_NATIVE_SOURCE}/third_party/ggml/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/jevtpp/licenses" RENAME ggml-LICENSE)
install(FILES "${PROJECT_SOURCE_DIR}/docs/licenses/nlohmann-json-LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/jevtpp/licenses")
