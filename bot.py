import sys
import dxcam
import numpy as np
import time
import cv2
import pyautogui
import keyboard
import threading
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import CenterCrop, Resize, Compose, ToTensor, Normalize
from PIL import Image
import ast
import glob

# Local imports
from train import GameNet

# Add global running flag
running = True

def check_for_exit():
    global running
    while True:
        if keyboard.is_pressed('esc'):
            print("ESC pressed, exiting...")
            running = False
            sys.exit()
        time.sleep(0.1)

def initialize_capture():
    # Start exit detection thread
    exit_thread = threading.Thread(target=check_for_exit, daemon=True)
    exit_thread.start()
    
    # Initialize DXcam
    camera = dxcam.create()
    
    # First, capture full screen to find the game window
    full_screen = camera.grab()
    # Capture entire desktop screen
    region = find_game_window(full_screen)

    return camera, region

keys = ["up", "left", "down", "right", "r"]
def get_pressed_keys():
    pressed_keys = []
    for i in range(len(keys)):
        pressed_keys.append(keyboard.is_pressed(keys[i]))
    if keyboard.is_pressed("w"):
        pressed_keys[0] = True
    if keyboard.is_pressed("space"):
        pressed_keys[0] = True
    if keyboard.is_pressed("s"):
        pressed_keys[2] = True
    if keyboard.is_pressed("a"):
        pressed_keys[1] = True
    if keyboard.is_pressed("d"):
        pressed_keys[3] = True
    return pressed_keys

def continuous_capture(camera, region, game, model=None):
    global running
    frames_since_last_input = 0
    last_keyboard_input = get_pressed_keys()
    last_replayed_input = None
    last_input_index = None
    training_run_name = "training_run_" + time.strftime("%Y%m%d_%H%M%S")
    while running:
        # Capture the specified region
        frame = camera.grab(region=region)
        
        if frame is not None:
            # Process the frame here
            # You can analyze the frame to determine snake position, food, obstacles, etc.
            
            # Example: Print the shape of the captured frame
            # print(f"Frame shape: {frame.shape}")
            # Get keyboard inputs
            keyboard_input = get_pressed_keys()
            if keyboard_input != last_keyboard_input:
                frames_since_last_input = 0
            else:
                frames_since_last_input += 1
            last_keyboard_input = keyboard_input

            # Downscale the image to 128x128
            # frame = cv2.resize(frame, (128, 128))

            if model is not None:
                # Run bot with provided model
                keyboard_input = predict_keys(model, frame)
                # print(f"Predicted keyboard input: {keyboard_input}")

                # Replay keyboard input
                max_index = 0
                for i in range(len(keyboard_input)):
                    # Press whichever input has the highest value
                    if keyboard_input[i] > keyboard_input[max_index]:
                        max_index = i

                if keyboard_input[max_index] > 0.1:
                    # If we are trying to move in the opposite direction as last time, move to the side instead
                    if last_replayed_input == keys[(max_index + 2) % 4] or last_replayed_input == keys[(max_index - 2) % 4]:
                        max_index = (max_index + 1) % 4
                    # Don't repeat inputs
                    # if keys[max_index] != last_replayed_input:
                    keyboard.press(keys[max_index])
                    last_replayed_input = keys[max_index]
                    if last_input_index != max_index and last_input_index is not None:
                        keyboard.release(keys[last_input_index])
                    last_input_index = max_index
                    # Print chosen key and percentages for each key
                    print(f"Pressed {keys[max_index]}, {keyboard_input[max_index]*100:.1f}%, " + 
                          ", ".join(f"{k}: {v*100:.1f}%" for k,v in zip(keys, keyboard_input)))
                else:
                    last_replayed_input = None
                if last_input_index is not None and keyboard_input[last_input_index] < 0.1:
                    keyboard.release(keys[last_input_index])
                    last_input_index = None
            else:
                # Write training images to file along with inputs
                if frames_since_last_input < 60:# and any(keyboard_input):
                    frame = cv2.resize(frame, (128, 128))
                    # If keyboard input contains "r", it's a restart - delete the last 10 seconds of training data
                    if keyboard_input[keys.index("r")]:
                        # Delete recent files in the training_data directory
                        for file in os.listdir(f"training_data/{game}/{training_run_name}"):
                            # Only remove files created in the last 10 seconds
                            file_path = f"training_data/{game}/{training_run_name}/{file}"
                            if time.time() - os.path.getctime(file_path) <= 10:
                                os.remove(file_path)
                                print(f"Deleted {file_path}")

                    print(f"Keyboard input: {keyboard_input}, Frames since last input: {frames_since_last_input}")
                    unixtimestamp = int(time.time() * 1000)
                    # Create training data directory if it doesn't exist
                    os.makedirs("training_data", exist_ok=True)
                    # Create directory for this training run if it doesn't exist
                    os.makedirs(f"training_data/{game}/{training_run_name}", exist_ok=True)
                    cv2.imwrite(f"training_data/{game}/{training_run_name}/frame_{unixtimestamp}.png", frame)
                    with open(f"training_data/{game}/{training_run_name}/inputs_{unixtimestamp}.txt", "w") as f:
                        f.write(str(keyboard_input))
            # cv2.imshow("Frame", frame)
            # # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
        # Add a small delay to prevent excessive CPU usage
        time.sleep(0.016)  # Approximately 60 FPS
    print("Capture stopped")
    os._exit(0)  # Force exit all threads

def find_game_window(frame):
    # Convert frame to HSV color space for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for your game (adjust these values)
    lower_bound = np.array([50, 50, 50])  # Example values
    upper_bound = np.array([255, 255, 255])
    
    # Create mask for game colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and show image for debugging
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # cv2.imshow("Contours", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    if contours:
        # Find the largest contour (assuming it's the game window)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, x + w, y + h)

    return None

def predict_keys(model, frame):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        Resize((480,480)),
        CenterCrop(480),
        Normalize(mean =[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225] )
    ])
    
    # Convert frame to PIL Image
    frame = Image.fromarray(frame)
    frame = transform(frame).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model.to(device)(frame)
    
    # Convert outputs to boolean values
    predictions = outputs.squeeze().cpu().numpy() #> 0.5
    return predictions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, required=True, help='Game to play')
    parser.add_argument('--mode', choices=['gather', 'bot'], required=True,
                       help='Run in data gathering mode or bot mode')
    args = parser.parse_args()

    camera, region = initialize_capture()
    if args.mode == 'gather':
        continuous_capture(camera, region, game=args.game)
    else:
        # Find most recent model
        models = sorted(glob.glob(f"training_data/{args.game}/training_run_*.pth"))
        if not models:
            print(f"No trained models found for {args.game}. Please train a model first.")
            sys.exit(1)
            
        latest_model = models[-1]
        print(f"Loading model: {latest_model}")
        
        model = GameNet()
        model.load_state_dict(torch.load(latest_model))
        model.eval()
        
        continuous_capture(camera, region, game=args.game, model=model)
