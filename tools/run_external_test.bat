@echo off
REM ===========================================================================
REM  Rev. 5 - cross-dataset test. No training; about 20-30 minutes on the GPU.
REM  Run from:  D:\PROGRAMMING\dr shimaa\code\lcms_db
REM  Do not run it while a training run is using the GPU.
REM   1. copies the Roboflow "birdrone" validation images to
REM      D:\PROGRAMMING\dr shimaa\External_birdrone  (class taken from the labels)
REM   2. checks that those images do not share scenes with the Kaggle training split
REM   3. scores the trained models on them with tools\evaluate_v2.py (same protocol)
REM   4. writes runs\EXTERNAL_TABLE.md
REM ===========================================================================
setlocal
set EXT=D:/PROGRAMMING/dr shimaa/External_birdrone
set IMG=%EXT%/images
set LBL=%EXT%/labels

echo [1/4] Preparing the external test set...
python tools\external_test.py prep --out "%EXT%" || goto :err

echo [2/4] Checking scene overlap with the Kaggle training split...
python tools\external_test.py check --out "%EXT%" --device cuda || goto :err

echo [3/4] Scoring the models on the external set...
python tools\evaluate_v2.py --config configs\final_full_budget.yaml --checkpoint runs\final_full_budget\best.pt --images "%IMG%" --labels "%LBL%" --out runs\final_full_budget\eval_external --device cuda || goto :err
python tools\evaluate_v2.py --config configs\A7_full_wiou_scaleaware.yaml --checkpoint runs\A7_full_wiou_scaleaware\best.pt --images "%IMG%" --labels "%LBL%" --out runs\A7_full_wiou_scaleaware\eval_external --device cuda || goto :err
for %%M in (yolov8s-scratch yolov8s yolov8n yolov8s-p2 yolo11s) do (
  python tools\evaluate_v2.py --yolo --checkpoint runs\baselines\%%M\weights\best.pt --images "%IMG%" --labels "%LBL%" --out runs\baselines\%%M_external --device cuda || goto :err
)

echo [4/4] Building the table...
python tools\collect_external.py || goto :err
echo.
echo Done. Results: runs\EXTERNAL_TABLE.md   Overlap check: %EXT%/overlap_with_kaggle_train.json
goto :eof
:err
echo FAILED at the step above. Fix and re-run this file.
exit /b 1
