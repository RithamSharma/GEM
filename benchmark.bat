@echo off
setlocal
REM ==========================================================================
REM  GEM  --  benchmark ONE netlist (throughput + Nsight Compute), no full rebuild
REM
REM      benchmark.bat  path\to\design.sv  top_module  [cycles]  [num_blocks]
REM
REM  Produces, in submission-results\ :
REM      partd_summary.txt         preserved-vs-shredded cycles/sec + graph size
REM      partd_preserved.json      per-rep timing samples (V2 heterogeneous)
REM      partd_shredded.json       per-rep timing samples (V1 baseline)
REM      part_b_v2_integrated.ncu-rep   Nsight Compute profile of the custom kernel
REM
REM  Requires WSL2 with the toolchain (same as compile.bat). Run compile.bat once
REM  first so the release binaries exist.
REM ==========================================================================

if "%~2"=="" (
    echo usage: benchmark.bat ^<design.sv^> ^<top_module^> [cycles] [num_blocks]
    exit /b 2
)

where wsl >nul 2>nul
if errorlevel 1 ( echo [FATAL] WSL2 not found - see compile.bat header. & exit /b 1 )

for /f "usebackq delims=" %%p in (`wsl wslpath -a "%~dp0." 2^>nul`) do set "REPO=%%p"
for /f "usebackq delims=" %%d in (`wsl wslpath -a "%~f1" 2^>nul`) do set "DES=%%d"

wsl bash -lc "cd '%REPO%' && chmod +x compile.sh scripts/*.sh && ./compile.sh --bench '%DES%' '%~2' '%~3' '%~4'"
exit /b %errorlevel%
