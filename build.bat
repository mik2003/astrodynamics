@echo off
cd /d "%~dp0"

echo ========================================
echo Building C++ Extension
echo ========================================

echo [1/4] Cleaning...
if exist build rmdir /s /q build 2>nul
if exist src\project\fast_module\_fast_module.pyd del src\project\fast_module\_fast_module.pyd 2>nul

echo [2/4] Configuring...
mkdir build 2>nul
cd build

cmake ../src/cpp -G "MinGW Makefiles"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: CMake failed!
    echo.
    echo Check that:
    echo 1. Python 3.13 is installed at C:\Python313
    echo 2. C:\Python313\libs\python313.lib exists
    echo 3. MinGW is in PATH
    echo.
    pause
    exit /b 1
)

echo [3/4] Building...
mingw32-make

if %errorlevel% neq 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo [4/4] Copying...
cd ..
copy build\_fast_module.pyd src\project\fast_module\ 2>nul

if exist src\project\fast_module\_fast_module.pyd (
    echo.
    echo SUCCESS! Built: src\project\fast_module\_fast_module.pyd
) else (
    echo ERROR: _fast_module.pyd not created!
    echo Check build\ directory...
    dir build
)

echo.
pause
