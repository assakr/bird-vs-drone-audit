@echo off
REM ===========================================================================
REM  LCMS-DB: every experiment the paper still needs, in order.
REM  Run from:  D:\PROGRAMMING\dr shimaa\code\lcms_db
REM  Needs:     CUDA GPU, the project's Python env, and  pip install ultralytics
REM  Approx. time on the GPU used for v2 (~126 s/epoch at 3000 samples/epoch):
REM    step 1 ~2 min | step 2 ~8 x 1.8 h | step 3 ~4 x 1-3 h | step 4 ~20+ h
REM  Each step can be run on its own; results land under runs\.
REM ===========================================================================
setlocal
set PYTHONUNBUFFERED=1
set DS=D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset
set CLEAN=D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset_clean

echo [1/4] Building clean split (valid/ bird labels are misaligned)...
python tools\make_clean_split.py --src "%DS%" --dst "%CLEAN%" || goto :err

echo [2/4] Ablation chain A0..A7 (same data, budget, seed; A7 = new scale-aware assigner)...
for %%C in (A0_plain A1_p2 A2_p2_msfe A3_p2_msfe_cbam A4_p2_msfe_cbam_bff A5_full_ciou A6_full_wiou A7_full_wiou_scaleaware) do (
  if not exist "runs\%%C\best.pt" (
    python train.py --config configs\%%C.yaml --device cuda:0 || goto :err
  )
  if not exist "runs\%%C\eval_v2_test\eval_v2.json" (
    python tools\evaluate_v2.py --config configs\%%C.yaml --checkpoint runs\%%C\best.pt ^
       --images "%CLEAN%/test/images" --labels "%CLEAN%/test/labels" --out runs\%%C\eval_v2_test --device cuda || goto :err
  )
)

echo [3/4] Standard baselines (YOLOv8n/s, YOLOv8s-P2, YOLO11s) on the same split...
python tools\train_baselines.py --data "%CLEAN%/data_clean.yaml" --epochs 50 || goto :err

echo Deciding target assignment for final_full_budget (A7 vs A6 on clean test split)...
python tools\check_and_prep_final.py || goto :err

echo [4/4] Final LCMS-DB with the full training set every epoch...
if not exist "runs\final_full_budget\best.pt" (
  python train.py --config configs\final_full_budget.yaml --device cuda:0 || goto :err
)
if not exist "runs\final_full_budget\eval_v2_test\eval_v2.json" (
  python tools\evaluate_v2.py --config configs\final_full_budget.yaml --checkpoint runs\final_full_budget\best.pt ^
     --images "%CLEAN%/test/images" --labels "%CLEAN%/test/labels" --out runs\final_full_budget\eval_v2_test --device cuda || goto :err
)

python tools\collect_results.py
echo Done. Tables: runs\RESULTS_TABLES.md
goto :eof
:err
echo FAILED at the step above. Fix and re-run that step only.
exit /b 1
