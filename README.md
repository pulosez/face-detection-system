# Face Detection System

## Prerequisites

- Python Version: 3.8.10+. More information haw to install Python for your operating system: https://www.python.org/downloads

### Create virtual environment

If you are not inside git repo root directory
```shell
$ cd path/to/the/repo
```
Create virtual environment
```shell
$ python3 -m venv venv
```
Activate virtual environment
```shell
$ source venv/bin/activate
```

### Install all packages

You need to be in the root directory
```shell
$ python3 -m pip install -r setup/requirements.txt
```

## System description

### Directories and modules description

- cfg: 
  - global_cfg.py - contains all global variables and logger initialize
- models:
  - face_detection_model: deep convolutional neural network for detecting faces on the input data
- setup:
  - requirements.txt - file listing all the dependencies
- workspace:
  - test_input_data: images for testing face detection system
  - detector.py - class for analyze image and find faces on it
  - detector_ui.py - user interface for face-detection-system

## How to run

Just run this command inside root directory
```shell
$ python3 workspace/detector_ui.py
```
