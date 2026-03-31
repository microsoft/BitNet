@echo off
REM ============================================================
REM Build bitnet.cpp for Windows x86_64 (Intel 12700K / AVX2)
REM ============================================================
REM
REM Prerequisites:
REM   1. Visual Studio 2022 with "Desktop development with C++" workload
REM   2. LLVM/Clang >= 18: https://github.com/llvm/llvm-project/releases
REM   3. CMake >= 3.22: https://cmake.org/download/
REM   4. Ninja: https://github.com/ninja-build/ninja/releases
REM   5. Python >= 3.9 with conda
REM
REM After installing VS 2022, run this from a "Developer Command Prompt for VS 2022"
REM or "x64 Native Tools Command Prompt for VS 2022"
REM
REM Usage:
REM   build_windows_12700k.bat
REM ============================================================

echo.
echo === BitNet TL2 Build for Intel 12700K (AVX2) ===
echo.

REM Check prerequisites
where clang >nul 2>&1
if errorlevel 1 (
    echo ERROR: clang not found. Install LLVM/Clang ^>= 18 and add to PATH.
    echo Download: https://github.com/llvm/llvm-project/releases
    exit /b 1
)

where cmake >nul 2>&1
if errorlevel 1 (
    echo ERROR: cmake not found. Install CMake ^>= 3.22.
    exit /b 1
)

where ninja >nul 2>&1
if errorlevel 1 (
    echo WARNING: ninja not found, will use default generator.
    echo For faster builds, install Ninja: https://github.com/ninja-build/ninja/releases
    set GENERATOR=-G "Visual Studio 17 2022" -T ClangCL
) else (
    set GENERATOR=-G Ninja
)

REM Initialize submodule if needed
if not exist "3rdparty\llama.cpp\CMakeLists.txt" (
    echo Initializing submodule...
    git submodule update --init
)

REM Apply the ggml NaN guard patch to the submodule
if exist "nan-guards-ggml.patch" (
    echo Applying NaN guard patch to llama.cpp submodule...
    pushd 3rdparty\llama.cpp
    git apply ..\..\nan-guards-ggml.patch 2>nul
    popd
)

REM Step 1: Generate TL2 kernels (uses Python)
echo.
echo === Step 1: Generating TL2 kernels ===
python utils\codegen_tl2.py --model bitnet_b1_58-2B --BM 256 --BK 96 --bm 32
if errorlevel 1 (
    echo WARNING: Kernel generation failed. Using existing kernels if available.
)

REM Step 2: CMake configure
echo.
echo === Step 2: CMake Configure ===
if not exist "build_win" mkdir build_win
cd build_win

cmake .. %GENERATOR% ^
    -DCMAKE_C_COMPILER=clang ^
    -DCMAKE_CXX_COMPILER=clang++ ^
    -DGGML_BITNET_X86_TL2=ON ^
    -DCMAKE_BUILD_TYPE=Release

if errorlevel 1 (
    echo ERROR: CMake configure failed.
    cd ..
    exit /b 1
)

REM Step 3: Build
echo.
echo === Step 3: Building ===
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

if errorlevel 1 (
    echo ERROR: Build failed.
    cd ..
    exit /b 1
)

cd ..

echo.
echo ============================================================
echo BUILD COMPLETE
echo ============================================================
echo.
echo Binaries are in: build_win\bin\Release\ (or build_win\bin\)
echo.
echo To run inference:
echo   build_win\bin\llama-cli.exe -m PATH\TO\bitnet_v2_TL2.gguf ^
echo       -p "What is an SBOM?" -n 200 --temp 0.7 -t 8
echo.
echo Interactive mode:
echo   build_win\bin\llama-cli.exe -m PATH\TO\bitnet_v2_TL2.gguf ^
echo       -i -p "<|im_start|>assistant" ^
echo       --in-prefix "<|im_start|>user\n" ^
echo       --in-suffix "<|im_end|>\n<|im_start|>assistant\n" ^
echo       -n 512 --temp 0.7 --repeat-penalty 1.2 -t 8 ^
echo       -r "<|im_end|>"
echo.
echo Server mode (access from any device):
echo   build_win\bin\llama-server.exe -m PATH\TO\bitnet_v2_TL2.gguf ^
echo       --host 0.0.0.0 --port 8080 -t 8 -c 4096
echo.
