from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy
import os


class JevtppConan(ConanFile):
    name = "jevtpp"
    version = "0.2.0"
    license = "MIT"
    url = "https://github.com/wiatrm/jevtpp"
    homepage = "https://github.com/wiatrm/jevtpp"
    description = "C++20 event matching, routing and evaluation with observable decisions"
    topics = ("routing", "rules-engine", "observability", "cpp20")
    package_type = "library"

    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "dashboard": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "dashboard": False,
    }

    exports_sources = (
        "CMakeLists.txt",
        "LICENSE",
        "cmake/*",
        "include/*",
        "src/*",
    )

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        toolchain = CMakeToolchain(self)
        toolchain.variables["JEVT_BUILD_TESTS"] = False
        toolchain.variables["JEVT_BUILD_BENCHMARKS"] = False
        toolchain.variables["JEVT_BUILD_EXAMPLES"] = False
        toolchain.variables["JEVT_ENABLE_HTTP"] = bool(self.options.dashboard)
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(self, "LICENSE", src=self.source_folder,
             dst=os.path.join(self.package_folder, "licenses"))
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "jevtpp")
        self.cpp_info.set_property("cmake_target_name", "jevt::jevt")
        self.cpp_info.libs = ["jevt"]

