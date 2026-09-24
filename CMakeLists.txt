cmake_minimum_required(VERSION 3.24)
project(JevTpp VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(JEVT_BUILD_TESTS "Build acceptance tests" ON)
option(JEVT_BUILD_BENCHMARKS "Build benchmark runner" ON)
option(JEVT_BUILD_EXAMPLES "Build examples" ON)
option(JEVT_ENABLE_HTTP "Build the optional diagnostics HTTP server" ON)

set(JEVT_SOURCES src/runtime.cpp src/diagnostics.cpp)
if(JEVT_ENABLE_HTTP)
  list(APPEND JEVT_SOURCES src/http_server.cpp)
endif()

add_library(jevt ${JEVT_SOURCES})
add_library(jevt::jevt ALIAS jevt)
target_compile_features(jevt PUBLIC cxx_std_20)
target_include_directories(jevt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(WIN32 AND JEVT_ENABLE_HTTP)
  target_link_libraries(jevt PRIVATE ws2_32)
endif()
set_target_properties(jevt PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0 EXPORT_NAME jevt)

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
