@echo off
REM run_all.bat - reruns every experiment in the paper, in order, then prints the numbers.
REM Double-click it, or run it from a terminal in this folder. Takes roughly 1.5-2 hours on CPU.
REM Each step's full output goes to runs\logs\<step>.txt; this window shows progress.

cd /d "%~dp0"

REM Write logs as UTF-8 (the model sometimes writes non-English characters)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM Where the model is. Use a local copy if you have one; otherwise this downloads it from
REM Hugging Face the first time (about 3 GB).
if "%QWEN_MODEL%"=="" set QWEN_MODEL=Qwen/Qwen2.5-1.5B-Instruct

if not exist runs\logs mkdir runs\logs
python -c "import sys, torch, transformers, numpy; print('python', sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__, '| numpy', numpy.__version__)" > runs\environment.txt
if errorlevel 1 goto :fail
python -m pip freeze > runs\pip_freeze.txt
type runs\environment.txt
echo Model: %QWEN_MODEL%

call :step collapse_open   python collapse_test.py --out runs\collapse_open || goto :stopped
call :step collapse_bf16   python collapse_test.py --dtype bfloat16 --out runs\collapse_bf16 || goto :stopped
call :step ablate_open     python ablate_test.py --out runs\ablate_open || goto :stopped
call :step multi           python wake_test.py --experiment multi --out runs\multi || goto :stopped
call :step wake            python wake_test.py --experiment wake --out runs\wake || goto :stopped
call :step sweep           python wake_test.py --experiment sweep --out runs\sweep || goto :stopped
call :step matched         python wake_test.py --experiment matched --compare-to runs\sweep --out runs\matched || goto :stopped
call :step rescaled        python wake_test.py --experiment rescaled --compare-to runs\sweep --out runs\rescaled || goto :stopped
call :step countdown_1000  python qwen_meditation.py --start 1000 --out runs\countdown_1000 || goto :stopped
call :step logit_lens      python logit_lens.py runs\countdown_1000 || goto :stopped
call :step countdown_100   python qwen_meditation.py --start 100 --out runs\countdown_100 || goto :stopped

python paper_numbers.py > runs\numbers.txt
echo.
echo All done %date% %time%. Every number in the paper is in runs\numbers.txt
pause
exit /b 0

:step
set NAME=%1
shift
set CMD=
:collect
if "%~1"=="" goto :run
set CMD=%CMD% %1
shift
goto :collect
:run
echo [%time%] %NAME% ...
%CMD% > runs\logs\%NAME%.txt 2>&1
if errorlevel 1 (
    echo [%time%] %NAME% FAILED - see runs\logs\%NAME%.txt
    exit /b 1
)
echo [%time%] %NAME% done
exit /b 0

:stopped
echo Stopped. Fix the step that failed, then run this again (finished steps are rerun too).
pause
exit /b 1

:fail
echo Python, torch, transformers or numpy is missing. Run:  python -m pip install -r requirements.txt
pause
exit /b 1
