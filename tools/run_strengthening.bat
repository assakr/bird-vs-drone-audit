@echo off
REM ===========================================================================
REM  Rev. 4 - experiments that make the paper's evidence solid.
REM  Run from:  D:\PROGRAMMING\dr shimaa\code\lcms_db
REM  Order: quick analyses first (steps 1-4, about 1-2 h) so you can review the
REM  audit pages while the multi-seed training (steps 5-6, about 25 h) runs.
REM  Each step can be run on its own; results land under runs\.
REM ===========================================================================
setlocal
set DS=D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset
set CLEAN=D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset_clean
set IMG=%CLEAN%/test/images
set LBL=%CLEAN%/test/labels
REM set to 1 to also repeat the 100-epoch final model twice (about 33 extra GPU-hours)
set RUN_FINAL_SEEDS=0

echo [1/6] Full label audit of train, valid and test (needs: pip install ultralytics)...
python tools\label_audit.py scan --dataset "%DS%" --weights runs\baselines\yolov8s-scratch\weights\best.pt --device 0 || goto :err

echo [2/6] Train/test scene overlap (SIFT voting + RANSAC)...
python tools\scene_overlap.py scan --dataset "%DS%" --device cuda || goto :err

echo [3/6] Saving ALL detections of the key models for bootstrap intervals...
python tools\evaluate_v2.py --config configs\final_full_budget.yaml --checkpoint runs\final_full_budget\best.pt --images "%IMG%" --labels "%LBL%" --out runs\final_full_budget\eval_all --keep-all-dets --device cuda || goto :err
python tools\evaluate_v2.py --config configs\A6_full_wiou.yaml --checkpoint runs\A6_full_wiou\best.pt --images "%IMG%" --labels "%LBL%" --out runs\A6_full_wiou\eval_all --keep-all-dets --device cuda || goto :err
python tools\evaluate_v2.py --config configs\A7_full_wiou_scaleaware.yaml --checkpoint runs\A7_full_wiou_scaleaware\best.pt --images "%IMG%" --labels "%LBL%" --out runs\A7_full_wiou_scaleaware\eval_all --keep-all-dets --device cuda || goto :err
for %%M in (yolov8s-scratch yolov8s yolov8n) do (
  python tools\evaluate_v2.py --yolo --checkpoint runs\baselines\%%M\weights\best.pt --images "%IMG%" --labels "%LBL%" --out runs\baselines\%%M_evalall --keep-all-dets --device cuda || goto :err
)

echo [4/6] Bootstrap 95%% confidence intervals...
python tools\bootstrap_ci.py --runs final=runs\final_full_budget\eval_all\per_image.json A6=runs\A6_full_wiou\eval_all\per_image.json A7=runs\A7_full_wiou_scaleaware\eval_all\per_image.json yolov8s-scratch=runs\baselines\yolov8s-scratch_evalall\per_image.json yolov8s=runs\baselines\yolov8s_evalall\per_image.json yolov8n=runs\baselines\yolov8n_evalall\per_image.json --compare A7:A6 yolov8s-scratch:final yolov8s:yolov8s-scratch --n 1000 --out runs\bootstrap_ci.json || goto :err

echo [5/6] Multi-seed: A5, A6, A7 with seeds 1 and 2 (about 11 h)...
for %%C in (A5_full_ciou A6_full_wiou A7_full_wiou_scaleaware) do (
  for %%S in (1 2) do (
    python train.py --config configs\%%C_s%%S.yaml --device cuda:0 || goto :err
    python tools\evaluate_v2.py --config configs\%%C_s%%S.yaml --checkpoint runs\%%C_s%%S\best.pt --images "%IMG%" --labels "%LBL%" --out runs\%%C_s%%S\eval_v2_test --device cuda || goto :err
  )
)

echo [6/6] Multi-seed: YOLOv8s from scratch with seeds 1 and 2 (about 14 h)...
for %%S in (1 2) do (
  python tools\train_baselines.py --data "%CLEAN%/data_clean.yaml" --models yolov8s-scratch --epochs 100 --seed %%S || goto :err
  python tools\evaluate_v2.py --yolo --checkpoint runs\baselines\yolov8s-scratch_s%%S\weights\best.pt --images "%IMG%" --labels "%LBL%" --out runs\baselines\yolov8s-scratch_s%%S_evalv2 --device cuda || goto :err
)
if "%RUN_FINAL_SEEDS%"=="1" (
  for %%S in (1 2) do (
    python train.py --config configs\final_full_budget_s%%S.yaml --device cuda:0 || goto :err
    python tools\evaluate_v2.py --config configs\final_full_budget_s%%S.yaml --checkpoint runs\final_full_budget_s%%S\best.pt --images "%IMG%" --labels "%LBL%" --out runs\final_full_budget_s%%S\eval_v2_test --device cuda || goto :err
  )
)
python tools\collect_seeds.py
echo.
echo Done. Now open runs\label_audit\review\review.html and runs\scene_overlap\review\review.html,
echo save both verdicts.csv files next to them, then run tools\run_after_review.bat
goto :eof
:err
echo FAILED at the step above. Fix and re-run that step only.
exit /b 1
