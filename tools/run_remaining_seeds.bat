@echo off
REM ===========================================================================
REM  Finishes the seed repeats of run_strengthening.bat without redoing finished
REM  ones: a run that already has eval_v2_test\eval_v2.json is skipped.
REM  Run from:  D:\PROGRAMMING\dr shimaa\code\lcms_db
REM ===========================================================================
setlocal
set CLEAN=D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset_clean
set IMG=%CLEAN%/test/images
set LBL=%CLEAN%/test/labels

for %%C in (A5_full_ciou_s1 A5_full_ciou_s2 A6_full_wiou_s1 A6_full_wiou_s2 A7_full_wiou_scaleaware_s1 A7_full_wiou_scaleaware_s2) do call :lcms %%C || goto :err
for %%S in (1 2) do call :yolo %%S || goto :err
python tools\collect_seeds.py
echo.
echo Done. Results are in runs\SEEDS_TABLE.md
goto :eof

:lcms
if exist runs\%1\eval_v2_test\eval_v2.json (
  echo [skip] %1 already evaluated
  exit /b 0
)
echo [train] %1
python train.py --config configs\%1.yaml --device cuda:0 || exit /b 1
python tools\evaluate_v2.py --config configs\%1.yaml --checkpoint runs\%1\best.pt --images "%IMG%" --labels "%LBL%" --out runs\%1\eval_v2_test --device cuda || exit /b 1
exit /b 0

:yolo
if exist runs\baselines\yolov8s-scratch_s%1_evalv2\eval_v2.json (
  echo [skip] yolov8s-scratch seed %1 already evaluated
  exit /b 0
)
echo [train] yolov8s-scratch seed %1
python tools\train_baselines.py --data "%CLEAN%/data_clean.yaml" --models yolov8s-scratch --epochs 100 --seed %1 || exit /b 1
python tools\evaluate_v2.py --yolo --checkpoint runs\baselines\yolov8s-scratch_s%1\weights\best.pt --images "%IMG%" --labels "%LBL%" --out runs\baselines\yolov8s-scratch_s%1_evalv2 --device cuda || exit /b 1
exit /b 0

:err
echo FAILED at the step above. Fix and re-run this file; finished runs are skipped.
exit /b 1
