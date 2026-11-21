@echo off
rem Builds a LibTorch/CUDA smoke tool: diagnose.exe <santorini_evaluator.pt> <position>
setlocal
if "%LIBTORCH%"=="" set "LIBTORCH=C:\Users\rafae\CLionProjects\santorini2025\extern\libtorch"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0"
if not exist ..\engine\build mkdir ..\engine\build
cl /nologo /std:c++20 /O2 /MD /EHsc /DNOMINMAX /W0 /I..\engine\src /I"%LIBTORCH%\include" /I"%LIBTORCH%\include\torch\csrc\api\include" ^
  diagnose.cpp ..\engine\src\board.cpp /Fo:..\engine\build\ /Fe:..\engine\build\diagnose.exe ^
  /link /INCLUDE:?warp_size@cuda@at@@YAHXZ /LIBPATH:"%LIBTORCH%\lib" torch.lib torch_cpu.lib c10.lib torch_cuda.lib c10_cuda.lib
exit /b %errorlevel%
