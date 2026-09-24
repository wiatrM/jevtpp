find_package(Threads REQUIRED)

add_executable(jevt_benchmark benchmark.cpp)
target_link_libraries(jevt_benchmark PRIVATE jevt::jevt Threads::Threads)
target_compile_features(jevt_benchmark PRIVATE cxx_std_20)

add_executable(jevt_benchmark_statistics_tests statistics_test.cpp)
target_compile_features(jevt_benchmark_statistics_tests PRIVATE cxx_std_20)

if(JEVT_BUILD_TESTS)
  add_test(NAME jevt_benchmark_statistics_tests COMMAND jevt_benchmark_statistics_tests)
  add_test(NAME jevt_benchmark_smoke
    COMMAND jevt_benchmark --iterations 1000 --warmup 100 --batch-size 8 --threads 2)
  set_tests_properties(jevt_benchmark_smoke PROPERTIES LABELS "benchmark-smoke")
endif()

if(JEVT_ENABLE_LAYA)
  add_executable(jevt_laya_latency laya_latency.cpp)
  target_link_libraries(jevt_laya_latency PRIVATE jevt::laya)
  target_compile_features(jevt_laya_latency PRIVATE cxx_std_20)
endif()

if(JEVT_ENABLE_LAYA AND JEVT_LAYA_MODEL_DIR)
  add_executable(jevt_laya_quality laya_quality.cpp)
  target_link_libraries(jevt_laya_quality PRIVATE jevt::laya)
  target_compile_features(jevt_laya_quality PRIVATE cxx_std_20)
endif()
