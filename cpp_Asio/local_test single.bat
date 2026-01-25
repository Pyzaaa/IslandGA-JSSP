@echo off
set EXE_NAME=ga_island_standaloneO2.exe
set INPUT_FILE=../data/N100M10.txt

:: Start Island 1 in a new window
:: It listens on 12345 and sends to 12346
start "Island A (Port 12345)" cmd /k "%EXE_NAME% %INPUT_FILE% %PARAMS% --my-port 12345 --peer-ip 127.0.0.1 --peer-port 12346 --pop 1000 --gen 1000 --mig-int 150 --verbose --chrono --init random"

echo [System] Both terminals started. 
echo [System] Look for the "MIGRATION REPORT" in the windows to see diversity metrics.
pause