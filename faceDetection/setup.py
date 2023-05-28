import sys
from cx_Freeze import setup, Executable

setup(name = "Simple Object Detection",
       version = "0.1",
       description = "software for object detection",
       executables = [Executable ("main.py")]
       )