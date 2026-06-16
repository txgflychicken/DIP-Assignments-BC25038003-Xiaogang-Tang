@echo off
REM COLMAP 3D reconstruction pipeline for Windows
REM Usage: run_colmap.bat

setlocal enabledelayedexpansion

set DATASET_PATH=data
set IMAGE_PATH=%DATASET_PATH%/images
set COLMAP_PATH=%DATASET_PATH%/colmap

if not exist "%COLMAP_PATH%\sparse" mkdir "%COLMAP_PATH%\sparse"
if not exist "%COLMAP_PATH%\dense" mkdir "%COLMAP_PATH%\dense"

echo === Step 1: Feature Extraction ===
colmap feature_extractor ^
    --database_path "%COLMAP_PATH%/database.db" ^
    --image_path "%IMAGE_PATH%" ^
    --ImageReader.camera_model PINHOLE ^
    --ImageReader.single_camera 1

echo === Step 2: Feature Matching ===
colmap exhaustive_matcher ^
    --database_path "%COLMAP_PATH%/database.db"

echo === Step 3: Sparse Reconstruction (Bundle Adjustment) ===
colmap mapper ^
    --database_path "%COLMAP_PATH%/database.db" ^
    --image_path "%IMAGE_PATH%" ^
    --output_path "%COLMAP_PATH%/sparse"

echo === Step 4: Image Undistortion ===
colmap image_undistorter ^
    --image_path "%IMAGE_PATH%" ^
    --input_path "%COLMAP_PATH%/sparse/0" ^
    --output_path "%COLMAP_PATH%/dense"

echo === Step 5: Dense Reconstruction (Patch Match Stereo) ===
colmap patch_match_stereo ^
    --workspace_path "%COLMAP_PATH%/dense"

echo === Step 6: Stereo Fusion ===
colmap stereo_fusion ^
    --workspace_path "%COLMAP_PATH%/dense" ^
    --output_path "%COLMAP_PATH%/dense/fused.ply"

echo === Done! ===
echo Results:
echo   Sparse: %COLMAP_PATH%/sparse/0/
echo   Dense:  %COLMAP_PATH%/dense/fused.ply