@echo off
rem Release build of Davi 3.0.0 Joker, exactly as the original release (MSVC x64 /O2 /MD).
rem Output: engine\build\Davi_3.0.0_Joker.exe. Set LIBTORCH to override the LibTorch root.
setlocal
if "%LIBTORCH%"=="" set "LIBTORCH=C:\Users\rafae\CLionProjects\santorini2025\extern\libtorch"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
cl /nologo /std:c++20 /O2 /MD /EHsc /DNOMINMAX /DNDEBUG /permissive- /bigobj /W0 ^
  /I"%LIBTORCH%\include" /I"%LIBTORCH%\include\torch\csrc\api\include" ^
  src\main.cpp src\board.cpp src\moves.cpp /Fo:build\ /Fe:build\Davi_3.0.0_Joker.exe ^
  /link /INCLUDE:?warp_size@cuda@at@@YAHXZ /LIBPATH:"%LIBTORCH%\lib" torch.lib torch_cpu.lib c10.lib torch_cuda.lib c10_cuda.lib
if errorlevel 1 exit /b 1
copy /y ..\weights\joker-3.0.0\santorini_evaluator.pt build\ >nul
exit /b 0
