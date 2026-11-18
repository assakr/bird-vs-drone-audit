@echo off
REM ===========================================================================
REM  LCMS-DB rev. 2 - next experiments (after run_all_experiments.bat)
REM  Run from:  D:\PROGRAMMING\dr shimaa\code\lcms_db
REM  Approx. GPU time:  step 1 up to ~13 h (early stopping may end it sooner)
REM                     step 2 ~7 h  | steps 3-4 ~15 min
REM  Each step can be run on its own; results land under runs\.
REM ===========================================================================
setlocal
set PYTHONUNBUFFERED=1
set CLEAN=D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset_clean
set IMG=%CLEAN%/test/images
set LBL=%CLEAN%/test/labels

echo [1/4] Finishing the final LCMS-DB run (stopped at epoch 16 of 100)...
REM  train.py --resume now restores the optimizer and continues the cosine LR schedule.
python train.py --config configs\final_full_budget.yaml --resume runs\final_full_budget\last.pt --device cuda:0 || goto :err
python tools\evaluate_v2.py --config configs\final_full_budget.yaml --checkpoint runs\final_full_budget\best.pt ^
   --images "%IMG%" --labels "%LBL%" --out runs\final_full_budget\eval_v2_test --device cuda || goto :err

echo [2/4] Fair baseline: YOLOv8s trained FROM SCRATCH (no COCO weights), 100 epochs...
python tools\train_baselines.py --data "%CLEAN%/data_clean.yaml" --models yolov8s-scratch --epochs 100 || goto :err

echo [3/4] Scoring every YOLO baseline with the SAME evaluator as LCMS-DB...
for %%M in (yolov8n yolov8s yolov8s-p2 yolo11s yolov8s-scratch) do (
  if exist runs\baselines\%%M\weights\best.pt (
    python tools\evaluate_v2.py --yolo --checkpoint runs\baselines\%%M\weights\best.pt ^
       --images "%IMG%" --labels "%LBL%" --out runs\baselines\%%M_evalv2 --device cuda || goto :err
  )
)

echo [4/4] Collecting tables...
python tools\collect_results.py
echo Done. Tables: runs\RESULTS_TABLES.md
goto :eof
:err
echo FAILED at the step above. Fix and re-run that step only.
exit /b 1
