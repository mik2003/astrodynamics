@echo off
cd /d "%~dp0"

:: Force English compiler output
set VSLANG=1033

echo ========================================
echo Building C++ Extension (_cpp_force_kernel)
echo ========================================

echo [1/4] Cleaning...
if exist build rmdir /s /q build 2>nul
if exist src\project\simulation\cpp_force_kernel\_cpp_force_kernel.pyd del src\project\simulation\cpp_force_kernel\_cpp_force_kernel.pyd 2>nul

echo [2/4] Configuring...
mkdir build 2>nul
cd build

cmake ../src/cpp -G "MinGW Makefiles"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: CMake failed!
    pause
    exit /b 1
)

echo [3/4] Building...
cmake --build . --config Release

if %errorlevel% neq 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo [4/4] Copying...
cd ..
copy build\Release\_cpp_force_kernel.pyd src\project\simulation\cpp_force_kernel\ 2>nul

if exist src\project\simulation\cpp_force_kernel\_cpp_force_kernel.pyd (
    echo.
    echo SUCCESS! Built: src\project\simulation\cpp_force_kernel\_cpp_force_kernel.pyd
) else (
    echo ERROR: _cpp_force_kernel.pyd not created!
    echo Check build\Release\ directory...
    dir build\Release
)

echo.
pause
