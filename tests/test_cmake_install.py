"""Exercise BitNet's install rules with a small llama.cpp dependency fixture."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class InstallTests(unittest.TestCase):
    def test_dependency_package_and_public_headers_survive_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, build, prefix = root / "source", root / "build", root / "prefix"
            source.mkdir()
            shutil.copyfile(ROOT / "CMakeLists.txt", source / "CMakeLists.txt")
            (source / "src").mkdir()
            (source / "src" / "CMakeLists.txt").write_text("")
            dependency = source / "3rdparty" / "llama.cpp"
            dependency.mkdir(parents=True)
            (dependency / "llama.c").write_text("int llama_fixture(void) { return 0; }\n")
            (dependency / "llama.h").write_text("int llama_fixture(void);\n")
            (dependency / "llama-cpp.h").write_text("#include <llama.h>\n")
            (dependency / "CMakeLists.txt").write_text('''
add_library(ggml INTERFACE)
add_library(llama STATIC llama.c)
set_target_properties(llama PROPERTIES PUBLIC_HEADER
    "${CMAKE_CURRENT_SOURCE_DIR}/llama.h;${CMAKE_CURRENT_SOURCE_DIR}/llama-cpp.h")
install(TARGETS llama ARCHIVE DESTINATION lib PUBLIC_HEADER DESTINATION include)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake" "# fixture package\\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake" DESTINATION lib/cmake/llama)
''')
            def run(*arguments):
                result = subprocess.run(["cmake", *map(str, arguments)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run("-S", source, "-B", build, "-DBUILD_NUMBER=1")
            run("--build", build)
            run("--install", build, "--prefix", prefix)
            self.assertTrue((prefix / "include" / "llama.h").is_file())
            self.assertTrue((prefix / "include" / "llama-cpp.h").is_file())
            self.assertTrue((prefix / "lib" / "cmake" / "llama" / "llama-config.cmake").is_file())


if __name__ == "__main__":
    unittest.main()
