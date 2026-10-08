@echo off
chcp 65001 >nul
echo =========================================================================
echo    CHAY BENCHMARK TOAN DIEN HE THONG (PHYSICAL + CYBER + CYBER-PHYSICAL)
echo =========================================================================
echo.
echo [1] Chay Quick-Test (Mac dinh - Mẫu rút gọn 5 giay)
echo [2] Chay Full Benchmark (Toan bo tap du lieu goc features/)
echo [3] Chay kem sinh bieu do truc quan (--plot)
echo [4] Thoat
echo.
set /p opt="Chon tuy chon [1-4] (Mac dinh 1): "

if "%opt%"=="2" (
    echo Dang chay Full Benchmark...
    python benchmark\run_all_benchmarks.py --full --plot
) else if "%opt%"=="3" (
    echo Dang chay kem sinh bieu do...
    python benchmark\run_all_benchmarks.py --plot
) else if "%opt%"=="4" (
    exit /b 0
) else (
    echo Dang chay Quick-Test...
    python benchmark\run_all_benchmarks.py --plot
)

echo.
echo Benchmark hoan tat! Kiem tra ket qua tai thu muc benchmark\results\
pause
