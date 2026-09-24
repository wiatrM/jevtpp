# Contributing to JevT++

Thank you for helping improve JevT++. Changes should be focused, tested and
portable across supported C++20 toolchains.

```sh
git clone https://github.com/wiatrm/jevtpp.git
cd jevtpp
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DJEVT_BUILD_TESTS=ON \
  -DJEVT_BUILD_EXAMPLES=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Pull requests should add a regression or acceptance test for behavior changes,
preserve the distinction between errors and abstention, keep metric labels
bounded, and avoid mandatory dependencies in the core.

For performance changes, include the exact command and environment. The
benchmark accepts `--iterations`, `--warmup`, `--batch-size`, `--threads` and
`--json`. Optional `--min-throughput` and `--max-p95-us` checks are intended
for controlled machines, not shared CI runners.

By contributing, you agree that your contribution is licensed under the MIT
License in this repository.

