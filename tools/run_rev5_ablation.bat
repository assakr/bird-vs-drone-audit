@echo off
REM ===========================================================================
REM  Rev. 5 - the two experiments the review asked for.
REM  Run from:  D:\PROGRAMMING\dr shimaa\code\lcms_db
REM   Part 1 (B0-B5, about 22 h): the architecture ablation repeated WITH the
REM           scale-aware assignment, so the module gains are measured on a
REM           detector that works.
REM   Part 2 (C1-C3, about 17 h): the scale-aware assignment with one of its three
REM           parts switched off at a time, to show which part gives the gain.
REM  Set RUN_PART2=0 to skip part 2. A finished run (best.pt present) is skipped,
REM  so the file can be re-started after an interruption.
REM ===========================================================================
setlocal
set CLEAN=D:/PROGRAMMING/dr shimaa/Bird vs Drone kaggel/Dataset_clean
set IMG=%CLEAN%/test/images
set LBL=%CLEAN%/test/labels
set RUN_PART2=1

for %%C in (B0_plain_sa B1_p2_sa B2_p2_msfe_sa B3_p2_msfe_cbam_sa B4_p2_msfe_cbam_bff_sa B5_full_ciou_sa) do call :one %%C || goto :err
if "%RUN_PART2%"=="1" (
  for %%C in (C1_sa_obj1 C2_sa_single_cell C3_sa_all_levels) do call :one %%C || goto :err
)
python tools\collect_rev5.py
echo.
echo Done. Results are in runs\REV5_TABLES.md
goto :eof

:one
if exist runs\%1\eval_v2_test\eval_v2.json (
  echo [skip] %1 already evaluated
  exit /b 0
)
echo [train] %1
python train.py --config configs\%1.yaml --device cuda:0 || exit /b 1
python tools\evaluate_v2.py --config configs\%1.yaml --checkpoint runs\%1\best.pt --images "%IMG%" --labels "%LBL%" --out runs\%1\eval_v2_test --device cuda || exit /b 1
exit /b 0

:err
echo FAILED at the step above. Fix and re-run this file; finished runs are skipped.
exit /b 1
