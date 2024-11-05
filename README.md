## Setup

- Install python 3.13.0
- Install micromamba `Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1).Content))`
- Install cuda toolkit https://developer.nvidia.com/cuda-downloads

Primary packages:

- Pytorch for ML stuff
- DXcam for screen capturing
- Pyautogui for mouse control

Other:

Whatever game you want to play.

Setup micromamba environment:

```
micromamba create -n bot -c pytorch -c conda-forge -c nvidia numpy matplotlib pytorch torchvision dxcam opencv pyqt pyautogui keyboard timm matplotlib
```

Remove environment:

```
micromamba remove -n bot --all
```

List environments:

```
micromamba env list
```

## Make pylance see packages in mamba environment

1. Ctrl + shift + p -> Python: Select Interpreter -> path -> `C:\Users\<username>\AppData\Roaming\mamba\envs\bot\python.exe`
2. Ctrl + shift + p -> Python: Restart language server

## Usage

There are 3 phases:

1. Data gathering

```
python bot.py --game <game> --mode gather
```

While in this mode, the script records your inputs and screenshots of the game. When you loose, press R to delete the last 10 seconds of data and keep playing. Press ESC to stop recording.

2. Model training

```
python train.py --game <game>
```

Trains a model with all available data.

3. Model playing

```
python bot.py --game <game> --mode bot
```

Plays the game using the trained model.
