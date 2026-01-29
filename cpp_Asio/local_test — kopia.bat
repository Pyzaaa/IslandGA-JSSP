@echo off
set EXE_NAME=Algorytm-GA-argumenty-Asio_island_FINAL.exe
set INPUT_FILE=../data/N1000M10.txt

:: Common GA Parameters
set PARAMS=--pop 5000 --stagnation 10000 --gen 5000 --verbose --chrono --init random

echo [System] Launching 2 Local Islands...

:: Start Island 1 in a new window
:: It listens on 12345 and sends to 12346
start "Island A (Port 12345)" cmd /k "%EXE_NAME% %INPUT_FILE% %PARAMS% --mig-int 1 --my-port 12345 --peer-ip 127.0.0.1 --peer-port 12346"

:: Start Island 2 in a new window
:: It listens on 12346 and sends to 12345
start "Island B (Port 12346)" cmd /k "%EXE_NAME% %INPUT_FILE% %PARAMS% --mig-int 150 --my-port 12346 --peer-ip 127.0.0.1 --peer-port 12345"

echo [System] Both terminals started. 
echo [System] Look for the "MIGRATION REPORT" in the windows to see diversity metrics.
pause