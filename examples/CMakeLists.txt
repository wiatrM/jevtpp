add_executable(jevt_support_routing support_routing.cpp)
target_link_libraries(jevt_support_routing PRIVATE jevt::jevt)
target_compile_features(jevt_support_routing PRIVATE cxx_std_20)

if(JEVT_ENABLE_REMOTE AND JEVT_REMOTE_CURL)
  add_executable(jevt_remote_routing remote_routing.cpp)
  target_link_libraries(jevt_remote_routing PRIVATE jevt::remote)
endif()

if(JEVT_ENABLE_LAYA)
  add_executable(jevt_laya_demo laya_routing_demo.cpp)
  target_link_libraries(jevt_laya_demo PRIVATE jevt::laya)
  target_compile_features(jevt_laya_demo PRIVATE cxx_std_20)
  if(JEVT_ENABLE_HTTP)
    target_compile_definitions(jevt_laya_demo PRIVATE JEVT_DEMO_HTTP=1)
  endif()
endif()

