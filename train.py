import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import CenterCrop, Resize, Compose, ToTensor, Normalize
from PIL import Image
import ast
import glob
import torchvision.models as models

transformer = Compose([
    Resize((480,480)),
    CenterCrop(480),
    Normalize(mean =[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225] )
])

class GameDataset(Dataset):
    def __init__(self, training_run_folder, game):
        # Get all training run folders
        training_runs = [d for d in glob.glob(f"training_data/{game}/training_run_*") if not d.endswith('.pth')]
        
        # Collect frame files from all runs
        self.frame_files = []
        for run_folder in training_runs:
            run_frames = sorted(glob.glob(f"{run_folder}/frame_*.png"))
            # Exclude first and last 60 frames from each run if it has enough frames
            frames_to_use = run_frames[60:-60] if len(run_frames) > 120 else []
            self.frame_files.extend(frames_to_use)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            Resize((480,480)),
            CenterCrop(480),
            Normalize(mean =[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225] )
        ])
        
    def __len__(self):
        return len(self.frame_files)
    
    def __getitem__(self, idx):
        frame_path = self.frame_files[idx]
        input_path = frame_path.replace('frame_', 'inputs_').replace('.png', '.txt')
        
        # Load image
        image = Image.open(frame_path)
        image = self.transform(image)
        
        # Load corresponding input
        with open(input_path, 'r') as f:
            inputs = ast.literal_eval(f.read())
            # If input length is 4, add a 0 at the end to make it 5
            if len(inputs) == 4:
                inputs.append(0)
        return image, torch.FloatTensor(inputs)
    
class GameNet(nn.Module):
    def __init__(self):
        super(GameNet, self).__init__()
        # Load pretrained EfficientNetV2 Small model
        efficientnet = models.efficientnet_v2_s(pretrained=True)
        
        # Remove the final classifier layer
        modules = list(efficientnet.children())[:-1]
        self.efficientnet = nn.Sequential(*modules)
        
        # Add our own classifier layer
        self.fc = nn.Linear(1280, 5)  # EfficientNetV2-S outputs 1280 features, we need 4 outputs for W,A,S,D
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet(x)
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc(x))
        return x
def train_model(training_run_folder, game, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    full_dataset = GameDataset(training_run_folder, game)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = GameNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        
        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{training_run_folder}.pth")
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
    
    return model

def dataset_statistics(training_run_folder, game):
    dataset = GameDataset(training_run_folder, game)
    print(f"Dataset size: {len(dataset)}")
    # Number of frames per input
    frames_per_input = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for item in dataset:
        input = item[1]
        frames_per_input[0] += input[0]
        frames_per_input[1] += input[1]
        frames_per_input[2] += input[2]
        frames_per_input[3] += input[3]
        frames_per_input[4] += input[4]
    print(f"Frames per input: {frames_per_input}")

# Add this to your main function:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, required=True, help='Game to train')
    args = parser.parse_args()

    # Find the most recent training run
    training_runs = [path for path in sorted(glob.glob(f"training_data/{args.game}/training_run_*")) if not path.endswith('.pth')]
    if training_runs:
        latest_run = training_runs[-1]
        dataset_statistics(latest_run, args.game)
        print(f"Training on data from: {latest_run}")
        train_model(latest_run, args.game)
