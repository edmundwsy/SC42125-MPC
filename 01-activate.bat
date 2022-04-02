@echo off
set SOURCE_DIR=%cd%

echo.
echo Adding %SOURCE_DIR%, each practicum dir and common dir to PYTHONPATH

set PYTHONPATH=%SOURCE_DIR%

rem activate conda environment h-mpc
conda activate h-mpc
