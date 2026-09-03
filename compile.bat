@echo off
setlocal EnableDelayedExpansion
REM ==========================================================================
REM  GEM heterogeneous-macro simulator  --  compile + verify + benchmark
REM
REM  Judges: double-click this file, or run it from a terminal:
REM      compile.bat              full build + correctness + throughput + Nsight
REM      compile.bat --quick      build + unit tests + functional gates only
REM      compile.bat --build-only just compile
REM
REM  All results are written to  submission-results\  in this folder.
REM
REM  The GEM toolchain (Yosys, cargo/nvcc, Icarus Verilog) is Linux-native, so
REM  this wrapper runs the build inside WSL2 -- NVIDIA's supported way to use
REM  CUDA from Windows.  One-time setup, from an elevated PowerShell:
REM
REM      wsl --install -d Ubuntu
REM      wsl -d Ubuntu -- sudo apt update ^&^& sudo apt install -y build-essential ^
REM          yosys iverilog python3 curl git
REM      wsl -d Ubuntu -- bash -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
REM      REM  install the CUDA toolkit for WSL2 (nvidia.com/cuda-downloads -> WSL-Ubuntu)
REM ==========================================================================

where wsl >nul 2>nul
if errorlevel 1 (
    echo [FATAL] WSL2 was not found on this machine.
    echo         Install it once with:  wsl --install -d Ubuntu
    echo         then reboot and run compile.bat again. See the header of this
    echo         file for the full one-time setup.
    exit /b 1
)

REM Resolve this script's folder to a path WSL understands.
set "WINDIR_HERE=%~dp0"
for /f "usebackq delims=" %%p in (`wsl wslpath -a "%WINDIR_HERE%." 2^>nul`) do set "REPO=%%p"
if "%REPO%"=="" (
    echo [FATAL] Could not translate "%WINDIR_HERE%" into a WSL path.
    exit /b 1
)

echo Running the GEM build inside WSL2:
echo     %REPO%
echo.
wsl bash -lc "cd '%REPO%' && sed -i 's/\r$//' compile.sh scripts/*.sh scripts/*.py 2>/dev/null; chmod +x compile.sh scripts/*.sh; ./compile.sh %*"
set RC=%errorlevel%

echo.
if "%RC%"=="0" (
    echo ==========================================================================
    echo  DONE.  Deliverable artifacts are in:  %WINDIR_HERE%submission-results\
    echo ==========================================================================
) else (
    echo [compile.bat] the build/verify run exited with code %RC% -- see
    echo               submission-results\compile.log
)
exit /b %RC%
