@echo off
setlocal
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set PYDIR=C:\Users\MIM.ASHRIF\AppData\Local\Python\pythoncore-3.14-64
set SCRIPTDIR=C:\Users\MIM.ASHRIF\Desktop\New folder (4)\Merge-Project
cd /d "%SCRIPTDIR%"
"%PYDIR%\python.exe" -u train_resnet50.py --fast > train_output.log 2>&1
