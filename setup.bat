@echo off
setlocal
REM ==========================================================================
REM  GEM  --  first-time setup on a fresh Windows machine
REM
REM  Step 1 (this script, run as Administrator):  install WSL2 + Ubuntu
REM  Step 2 (after the reboot it asks for):       double-click setup.bat again
REM                                               -> installs the Linux toolchain
REM  Step 3:                                      compile.bat
REM
REM  If you are already on Linux, just run:  bash scripts/install_deps.sh
REM ==========================================================================

where wsl >nul 2>nul
if errorlevel 1 goto INSTALL_WSL

REM WSL is present -- check for a distro, then install the toolchain inside it.
wsl -l -q >nul 2>nul
if errorlevel 1 goto INSTALL_WSL

for /f "usebackq delims=" %%p in (`wsl wslpath -a "%~dp0." 2^>nul`) do set "REPO=%%p"
echo Installing the GEM toolchain inside WSL (Rust, Yosys 0.68, Icarus, ...) ...
echo (CUDA Toolkit is a separate NVIDIA download -- the script prints the steps.)
echo.
wsl bash -lc "cd '%REPO%' && sed -i 's/\r$//' scripts/*.sh && bash scripts/install_deps.sh"
echo.
echo ==========================================================================
echo  Toolchain install finished.  Next:  compile.bat
echo  (CUDA: follow the instructions install_deps.sh printed, if nvcc was missing)
echo ==========================================================================
exit /b 0

:INSTALL_WSL
echo Installing WSL2 + Ubuntu.  This needs Administrator rights and a reboot.
echo.
net session >nul 2>nul
if errorlevel 1 (
    echo   Right-click setup.bat  ->  "Run as administrator", then try again.
    exit /b 1
)
wsl --install -d Ubuntu
echo.
echo ==========================================================================
echo  REBOOT now.  After the reboot, Ubuntu will finish its first-run setup
echo  (pick a username + password), then run setup.bat again to install the
echo  build toolchain.
echo ==========================================================================
exit /b 0
