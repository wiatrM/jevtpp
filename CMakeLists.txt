cmake_minimum_required(VERSION 3.24)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
project(JevTpp VERSION 0.2.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)

option(JEVT_BUILD_TESTS "Build acceptance tests" ${PROJECT_IS_TOP_LEVEL})
option(JEVT_BUILD_BENCHMARKS "Build benchmark runner" ${PROJECT_IS_TOP_LEVEL})
option(JEVT_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOP_LEVEL})
option(JEVT_ENABLE_HTTP "Build the optional diagnostics HTTP server" ON)
option(JEVT_ENABLE_REMOTE "Build the optional remote HTTP backend" OFF)
option(JEVT_ENABLE_ASIO "Expose the optional Boost.Asio integration" OFF)
option(JEVT_ENABLE_LAYA "Build the native ONNX Runtime Laya adapter" OFF)
option(JEVT_ENABLE_LAYA_NATIVE "Build the optional laya.cpp/ggml adapter" OFF)
option(JEVT_LAYA_NATIVE_CUDA "Enable CUDA in the laya.cpp adapter" OFF)
set(JEVT_LAYA_NATIVE_SOURCE_DIR "" CACHE PATH "Optional existing pinned laya.cpp source with submodules")
set(JEVT_LAYA_NATIVE_MODEL_DIR "" CACHE PATH "Safetensors bundle for native integration tests")
option(JEVT_FETCH_TOKENIZERS_CPP "Fetch the pinned tokenizers-cpp dependency" ON)
set(JEVT_LAYA_MODEL_DIR "" CACHE PATH "Model bundle used by the opt-in Laya integration test")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(JEVT_SOURCES src/runtime.cpp src/diagnostics.cpp src/batching.cpp)
if(JEVT_ENABLE_HTTP)
  list(APPEND JEVT_SOURCES src/http_server.cpp)
endif()

add_library(jevt ${JEVT_SOURCES})
add_library(jevt::jevt ALIAS jevt)
target_link_libraries(jevt PUBLIC Threads::Threads)
target_compile_features(jevt PUBLIC cxx_std_20)
target_include_directories(jevt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(WIN32 AND JEVT_ENABLE_HTTP)
  target_link_libraries(jevt PRIVATE ws2_32)
endif()
set_target_properties(jevt PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0 EXPORT_NAME jevt
  WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(JEVT_ENABLE_LAYA)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set_target_properties(jevt PROPERTIES POSITION_INDEPENDENT_CODE ON)
  find_package(onnxruntime REQUIRED)
  if(NOT TARGET tokenizers_cpp)
    if(NOT JEVT_FETCH_TOKENIZERS_CPP)
      find_package(tokenizers_cpp CONFIG REQUIRED)
    else()
      include(FetchContent)
      set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
      FetchContent_Declare(tokenizers_cpp
        GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
        GIT_TAG c586c52f93f7b060753bd2388eb96a105cb7374d
        GIT_SHALLOW FALSE)
      # The vendored dependency has broad install() rules (including
      # sentencepiece/msgpack). EXCLUDE_FROM_ALL keeps JevT++ installs clean.
      FetchContent_GetProperties(tokenizers_cpp)
      if(NOT tokenizers_cpp_POPULATED)
        FetchContent_Populate(tokenizers_cpp)
        add_subdirectory("${tokenizers_cpp_SOURCE_DIR}" "${tokenizers_cpp_BINARY_DIR}" EXCLUDE_FROM_ALL)
      endif()
    endif()
  endif()
  add_library(jevt_laya SHARED src/laya.cpp)
  add_library(jevt::laya ALIAS jevt_laya)
  target_compile_features(jevt_laya PUBLIC cxx_std_20)
  target_include_directories(jevt_laya PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(jevt_laya PUBLIC jevt::jevt PRIVATE onnxruntime::onnxruntime tokenizers_cpp)
  set_target_properties(jevt_laya PROPERTIES
    VERSION ${PROJECT_VERSION} SOVERSION 0 EXPORT_NAME laya WINDOWS_EXPORT_ALL_SYMBOLS ON)
  install(TARGETS jevt_laya EXPORT JevTTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(JEVT_ENABLE_LAYA_NATIVE)
  include(cmake/LayaNative.cmake)
endif()

if(JEVT_ENABLE_REMOTE)
  include(cmake/Remote.cmake)
endif()

if(JEVT_ENABLE_ASIO)
  find_package(Boost 1.74 REQUIRED)
  add_library(jevt_asio INTERFACE)
  add_library(jevt::asio ALIAS jevt_asio)
  target_link_libraries(jevt_asio INTERFACE jevt::jevt Boost::headers)
  set_target_properties(jevt_asio PROPERTIES EXPORT_NAME asio)
  install(TARGETS jevt_asio EXPORT JevTTargets)
endif()

if(MSVC)
  target_compile_options(jevt PRIVATE /W4 /permissive-)
else()
  target_compile_options(jevt PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(JEVT_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
  add_subdirectory(examples)
endif()
if(JEVT_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  enable_testing()
  add_subdirectory(tests)
endif()
if(JEVT_BUILD_BENCHMARKS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
  add_subdirectory(benchmarks)
endif()

install(TARGETS jevt EXPORT JevTTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT JevTTargets FILE jevtppTargets.cmake NAMESPACE jevt:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jevtpp)

configure_package_config_file(cmake/jevtppConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/jevtppConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jevtpp)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/jevtppConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/jevtppConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/jevtppConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jevtpp)
