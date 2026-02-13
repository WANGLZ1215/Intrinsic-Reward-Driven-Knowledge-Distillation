@echo off
REM Batch evaluation of all RL checkpoints (Windows version)

REM Configuration
set CHECKPOINT_DIR=checkpoints\rl_model
set OUTPUT_DIR=evaluation_results
set EVAL_SAMPLES=100

REM Create output directory
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Evaluate all checkpoints
echo Starting batch checkpoint evaluation...
echo Checkpoint directory: %CHECKPOINT_DIR%
echo Output directory: %OUTPUT_DIR%
echo Evaluation samples: %EVAL_SAMPLES%
echo.

REM Iterate through all checkpoint directories
for /d %%d in (%CHECKPOINT_DIR%\checkpoint-*) do (
    if exist "%%d" (
        echo ==========================================
        echo Evaluating checkpoint: %%~nxd
        echo ==========================================
        
        set output_file=%OUTPUT_DIR%\evaluation_results_%%~nxd.json
        
        REM Execute evaluation
        python evaluation\evaluate_checkpoint.py ^
            --checkpoint_path "%%d" ^
            --eval_samples %EVAL_SAMPLES% ^
            --output_file "%output_file%"
        
        if %errorlevel% equ 0 (
            echo [Success] %%~nxd evaluation completed
        ) else (
            echo [Failed] %%~nxd evaluation failed
        )
        echo.
    )
)

echo ==========================================
echo Batch evaluation completed!
echo Results saved in: %OUTPUT_DIR%
echo ==========================================
pause

