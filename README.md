# ANPR Live Camera Prototype

Prototype for detecting vehicle license plates and reading plate text (OCR) from images or a live camera.

## Requirements

- Python 3.10+ (recommended; confirm compatibility with your installed FastALPR version)
- A webcam for live-camera mode
- Internet access on first run if the model weights need to be downloaded automatically

## Setup

Create and activate a virtual environment:

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

### Live camera

If `live_camera.py` is included in the repository:

```powershell
python live_camera.py
```

### Run the main program

Check the available arguments in `anpr_ev.py`. For example, if the script supports webcam index 0:

```powershell
python anpr_ev.py --video 0
```

The exact command depends on the argument parser in the script. Do not assume this example works until it has been tested with the committed version.

## Testing with static images

The project was initially tested using static plate images before moving to live-camera testing. Static-image testing makes it easier to compare predicted text with manually verified ground truth and identify OCR errors under more controlled conditions.

If you publish test images or spreadsheets, confirm that you have permission to share them and remove sensitive information. Report performance metrics only after defining what counts as a correct prediction and comparing predictions against ground truth.

## How it works

1. Read an image or camera frame.
2. Detect the license-plate region.
3. Recognize the characters in the plate region using OCR.
4. Display or record the predicted plate text.

Detection of the plate region and correct recognition of every character are separate outcomes.

## Notes

- This repository contains a prototype, not a guarantee of production-ready accuracy.
- Model weights may be downloaded by the library on first use; confirm this behavior for the installed FastALPR version.
- Do not upload the `venv/` directory.
- Do not upload private images, credentials, API keys, or datasets unless you have permission and intend to publish them.
