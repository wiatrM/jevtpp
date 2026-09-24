option(JEVT_REMOTE_CURL "Build the remote backend's default libcurl transport" ON)

if(NOT TARGET nlohmann_json::nlohmann_json)
  find_package(nlohmann_json 3.11 QUIET CONFIG)
endif()
if(NOT TARGET nlohmann_json::nlohmann_json)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG 55f93686c01528224f448c19128836e7df245f72
    GIT_SHALLOW FALSE)
  FetchContent_GetProperties(nlohmann_json)
  if(NOT nlohmann_json_POPULATED)
    FetchContent_Populate(nlohmann_json)
    # Keep dependency examples, tests and install rules out of our package.
    add_subdirectory("${nlohmann_json_SOURCE_DIR}" "${nlohmann_json_BINARY_DIR}" EXCLUDE_FROM_ALL)
  endif()
endif()

set_target_properties(jevt PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(jevt_remote SHARED src/remote.cpp)
add_library(jevt::remote ALIAS jevt_remote)
target_link_libraries(jevt_remote PUBLIC jevt::jevt PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(jevt_remote PUBLIC cxx_std_20)
if(MSVC)
  target_compile_options(jevt_remote PRIVATE /W4 /permissive-)
else()
  target_compile_options(jevt_remote PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(JEVT_REMOTE_CURL)
  find_package(CURL 7.66 REQUIRED)
  target_sources(jevt_remote PRIVATE src/curl_transport.cpp)
  target_link_libraries(jevt_remote PRIVATE CURL::libcurl)
  target_compile_definitions(jevt_remote PRIVATE JEVT_REMOTE_CURL)
endif()
set_target_properties(jevt_remote PROPERTIES
  VERSION ${PROJECT_VERSION} SOVERSION 0 EXPORT_NAME remote WINDOWS_EXPORT_ALL_SYMBOLS ON)
install(TARGETS jevt_remote EXPORT JevTTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES "${PROJECT_SOURCE_DIR}/docs/licenses/nlohmann-json-LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/jevtpp/licenses")
