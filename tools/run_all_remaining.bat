@echo off
setlocal
set CLEAN=D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset_clean
set IMG=%CLEAN%/test/images
set LBL=%CLEAN%/test/labels

echo ===================================================================
echo [1/3] Multi-seed: A5, A6, A7 with seeds 1 and 2
echo ===================================================================
for %%C in (A5_full_ciou A6_full_wiou A7_full_wiou_scaleaware) do (
  for %%S in (1 2) do (
    call :train_lcms %%C_s%%S || goto :err
  )
)

echo ===================================================================
echo [2/3] Multi-seed: YOLOv8s from scratch with seeds 1 and 2
echo ===================================================================
for %%S in (1 2) do (
  call :train_yolo yolov8s-scratch %%S || goto :err
)

python tools\collect_seeds.py

echo ===================================================================
echo [3/3] Rev. 5 Ablation Study (B0-B5 and C1-C3)
echo ===================================================================
for %%C in (B0_plain_sa B1_p2_sa B2_p2_msfe_sa B3_p2_msfe_cbam_sa B4_p2_msfe_cbam_bff_sa B5_full_ciou_sa C1_sa_obj1 C2_sa_single_cell C3_sa_all_levels) do (
  call :train_lcms %%C || goto :err
)

python tools\collect_rev5.py
python tools\collect_results.py

echo.
echo ===================================================================
echo All multi-seed and Rev. 5 ablation experiments completed successfully!
echo ===================================================================
goto :eof

:train_lcms
if exist runs\%1\eval_v2_test\eval_v2.json (
  echo [skip] %1 already trained and evaluated
  exit /b 0
)
if exist runs\%1\best.pt (
  echo [eval] %1 checkpoint exists, running evaluate_v2...
  python tools\evaluate_v2.py --config configs\%1.yaml --checkpoint runs\%1\best.pt --images "%IMG%" --labels "%LBL%" --out runs\%1\eval_v2_test --device cuda || exit /b 1
  exit /b 0
)
echo [train] %1
python train.py --config configs\%1.yaml --device cuda:0 || exit /b 1
python tools\evaluate_v2.py --config configs\%1.yaml --checkpoint runs\%1\best.pt --images "%IMG%" --labels "%LBL%" --out runs\%1\eval_v2_test --device cuda || exit /b 1
exit /b 0

:train_yolo
if exist runs\baselines\%1_s%2_evalv2\eval_v2.json (
  echo [skip] %1_s%2 already evaluated
  exit /b 0
)
if not exist runs\baselines\%1_s%2\weights\best.pt (
  echo [train] YOLO %1 seed %2
  python tools\train_baselines.py --data "%CLEAN%/data_clean.yaml" --models %1 --epochs 100 --seed %2 || exit /b 1
)
python tools\evaluate_v2.py --yolo --checkpoint runs\baselines\%1_s%2\weights\best.pt --images "%IMG%" --labels "%LBL%" --out runs\baselines\%1_s%2_evalv2 --device cuda || exit /b 1
exit /b 0

:err
echo FAILED during execution. Fix and re-run.
exit /b 1
