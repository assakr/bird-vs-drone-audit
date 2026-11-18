@echo off
REM ===========================================================================
REM  Rev. 4 - run AFTER saving both verdicts.csv files:
REM    runs\label_audit\review\verdicts.csv
REM    runs\scene_overlap\review\verdicts.csv
REM  Run from:  D:\PROGRAMMING\dr shimaa\code\lcms_db      (about 15 min)
REM ===========================================================================
setlocal
set CLEAN=D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset_clean
set IMG=%CLEAN%/test/images
set LBL=%CLEAN%/test/labels
set LIST=runs\scene_overlap\clean_test_images.txt

echo [1/3] Label error rates per split...
python tools\label_audit.py summarize || goto :err

echo [2/3] Scene overlap estimate and the leakage-free test list...
python tools\scene_overlap.py summarize || goto :err

echo [3/3] Re-scoring models on test images with no same-scene training image...
python tools\evaluate_v2.py --config configs\final_full_budget.yaml --checkpoint runs\final_full_budget\best.pt --images "%IMG%" --labels "%LBL%" --image-list %LIST% --out runs\final_full_budget\eval_unseen --device cuda || goto :err
python tools\evaluate_v2.py --config configs\A7_full_wiou_scaleaware.yaml --checkpoint runs\A7_full_wiou_scaleaware\best.pt --images "%IMG%" --labels "%LBL%" --image-list %LIST% --out runs\A7_full_wiou_scaleaware\eval_unseen --device cuda || goto :err
for %%M in (yolov8s-scratch yolov8s yolov8n yolov8s-p2 yolo11s) do (
  python tools\evaluate_v2.py --yolo --checkpoint runs\baselines\%%M\weights\best.pt --images "%IMG%" --labels "%LBL%" --image-list %LIST% --out runs\baselines\%%M_unseen --device cuda || goto :err
)
python tools\collect_results.py
echo Done. Send runs\RESULTS_TABLES.md, runs\SEEDS_TABLE.md, runs\bootstrap_ci.json,
echo runs\label_audit\audit_estimate.json and runs\scene_overlap\overlap_estimate.json back for the paper.
goto :eof
:err
echo FAILED at the step above. Fix and re-run that step only.
exit /b 1
