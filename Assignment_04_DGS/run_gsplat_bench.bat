@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d E:\RCALS\数字图像处理作业\Assignment_04_DGS
python -u compare_with_gsplat.py
