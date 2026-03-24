# gpu_check.py
from numba import cuda

print("GPU available:", cuda.is_available())

if cuda.is_available():
    print("Device:", cuda.get_current_device().name)