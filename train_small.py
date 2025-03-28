import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_len=2500):
        super().__init__()
        if not hasattr(self, "pe"):
            pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]

            pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
            pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

            pe = pe.unsqueeze(0).transpose(1, 2)  # [1, d_model, max_len]
            self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2)]

class CNNEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, d_model=64, nhead=8, d_hid=64, nlayers=2, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=MAP_SIZE**2)
        self.cnn_encoder = CNNEncoding()
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.cnn_encoder(src)
        src = self.pos_encoder(src)
        src = torch.transpose(src, 1, 2)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        # print(f"Transformer Output: Min={output.min().item()}, Max={output.max().item()}")
        return output

class SimpleDataset():
    def __init__(self, x_file, y_file, batch_size=20, train=True):
        inputs = np.load(x_file)["arr_0"] 
        labels = np.load(y_file)["arr_0"]

        if np.isnan(inputs).any() or np.isnan(labels).any():
            raise ValueError("Dataset contains NaN values!")
        if np.isinf(inputs).any() or np.isinf(labels).any():
            raise ValueError("Dataset contains Inf values!")

        # Adjust the channel dimension of inputs if necessary
        if inputs.shape[1] != 2:
            raise ValueError(f"Inputs must have 2 channels (map and goal), but got {inputs.shape[1]} channels.")
        
        self.inputs = torch.Tensor(inputs).float()  # Shape: [batch_size, 2, MAP_SIZE, MAP_SIZE]
        self.labels = torch.Tensor(labels).float()  # Shape: [batch_size, MAP_SIZE, MAP_SIZE]
        self.batch_size = batch_size
        self.train = train

    def get_dataloader(self):
        dataset = TensorDataset(self.inputs, self.labels)
        sampler = RandomSampler(dataset) if self.train else SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)


class WrappedModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.model = TransformerModel()
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Reshape y_hat to match the shape of y: [batch_size, MAP_SIZE, MAP_SIZE]
        y_hat = y_hat.view(y.size(0), MAP_SIZE, MAP_SIZE)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Reshape y_hat to match the shape of y: [batch_size, MAP_SIZE, MAP_SIZE]
        y_hat = y_hat.view(y.size(0), MAP_SIZE, MAP_SIZE)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def visualize_output(model_checkpoint, input_data_path, output_data_path, index=0, map_size=15, save_dir="map_result"):
    # Load the trained model
    model = WrappedModel.load_from_checkpoint(model_checkpoint)
    model.eval()

    # Load input and ground truth data
    try:
        inputs = np.load(input_data_path)["arr_0"]  # Shape: [size, 2, map_size, map_size]
        ground_truths = np.load(output_data_path)["arr_0"]  # Shape: [size, map_size, map_size]
    except KeyError as e:
        print(f"Error: Missing expected array in file: {e}")
        return
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return

    if index >= inputs.shape[0]:
        print(f"Error: Index {index} is out of bounds for the dataset.")
        return

    input_sample = torch.Tensor(inputs[index]).unsqueeze(0)  # Add batch dimension, shape: [1, 2, map_size, map_size]
    ground_truth = ground_truths[index]  # True cost map, shape: [map_size, map_size]

    # Generate prediction
    with torch.no_grad():
        output = model(input_sample)  # Shape: [1, map_size**2, 1]
        output = output.view(map_size, map_size).numpy()  # Reshape to [map_size, map_size]

    obstacle_map = inputs[index, 0]  
    goal_map = inputs[index, 1]  

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.title("Obstacle Map")
    plt.imshow(obstacle_map, cmap="viridis", vmin=0, vmax=5)
    plt.colorbar(label="Difficulty Scale")

    plt.subplot(1, 4, 2)
    plt.title("Goal Map")
    plt.imshow(goal_map, cmap="binary")
    plt.colorbar(label="Goal Presence")

    plt.subplot(1, 4, 3)
    plt.title("Model Output")
    plt.imshow(output, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Predicted Cost")

    plt.subplot(1, 4, 4)
    plt.title("Ground Truth")
    plt.imshow(ground_truth, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="True Cost")

    plt.tight_layout()
 
    os.makedirs(save_dir, exist_ok=True)

    # Save the result while avoiding duplicates
    save_path = os.path.join(save_dir, f"result_{index}.png")
    counter = 1
    while os.path.exists(save_path):  # Check for duplicate filenames
        save_path = os.path.join(save_dir, f"result_{index}_{counter}.png")
        counter += 1

    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()
    plt.close()


def compute_map_accuracy(model_checkpoint, input_data_path, output_data_path, index=0, map_size=15):
    model = WrappedModel.load_from_checkpoint(model_checkpoint)
    model.eval()

    # Load input and ground truth data
    inputs = np.load(input_data_path)["arr_0"]  # Shape: [size, 2, map_size, map_size]
    ground_truths = np.load(output_data_path)["arr_0"]  # Shape: [size, map_size, map_size]

    input_sample = torch.Tensor(inputs[index]).unsqueeze(0)  # Add batch dimension
    ground_truth = ground_truths[index]  # True cost map

    with torch.no_grad():
        output = model(input_sample)  # Shape: [1, map_size**2, 1]
        output = output.view(map_size, map_size).numpy()  # Reshape to [map_size, map_size]

    # Normalize values for comparison
    max_gt = np.max(ground_truth) if np.max(ground_truth) > 0 else 1
    max_pred = np.max(output) if np.max(output) > 0 else 1

    norm_ground_truth = ground_truth / max_gt  # Normalize between 0-1
    norm_output = output / max_pred  # Normalize between 0-1

    # Generate prediction
    with torch.no_grad():
        output = model(input_sample)  # Shape: [1, map_size**2, 1]
        output = output.view(map_size, map_size).numpy()  # Reshape to [map_size, map_size]

    if np.issubdtype(ground_truth.dtype, np.integer):
        norm_output = np.round(norm_output * max_gt)

    tolerance = 0.05  # Allow a small 5% difference
    correct_pixels = np.abs(norm_output - norm_ground_truth) < tolerance 
    accuracy = np.mean(correct_pixels) * 100 

    print(f"Accuracy for map generation {index}: {accuracy:.2f}%")
    return accuracy

def get_neighbors(x, y, map_size):
    """Get the coordinates of neighboring nodes within the map boundaries."""
    neighbors = []
    if x > 0:         # Up
        neighbors.append((x - 1, y))
    if x < map_size - 1:  # Down
        neighbors.append((x + 1, y))
    if y > 0:         # Left
        neighbors.append((x, y - 1))
    if y < map_size - 1:  # Right
        neighbors.append((x, y + 1))
    return neighbors

def get_best_directions(cost_map, x, y, map_size):
    """Returns the best directions (minimum cost) to move from a given node."""
    neighbors = get_neighbors(x, y, map_size)
    neighbor_costs = {neighbor: cost_map[neighbor] for neighbor in neighbors}

    min_cost = min(neighbor_costs.values())  # Identify the minimum cost among neighbors
    best_directions = [direction for direction, cost in neighbor_costs.items() if cost == min_cost]

    return best_directions

def compute_accuracy_path(model_checkpoint, input_data_path, output_data_path, index=0, map_size=15):
    model = WrappedModel.load_from_checkpoint(model_checkpoint)
    model.eval()

    inputs = np.load(input_data_path)["arr_0"]  # Shape: [size, 2, map_size, map_size]
    ground_truths = np.load(output_data_path)["arr_0"]  # Shape: [size, map_size, map_size]

    input_sample = torch.Tensor(inputs[index]).unsqueeze(0)  # Add batch dimension
    ground_truth = ground_truths[index]  # True cost map

    with torch.no_grad():
        output = model(input_sample)  # Shape: [1, map_size**2, 1]
        output = output.view(map_size, map_size).numpy()  # Reshape to [map_size, map_size]

    correct_moves = 0
    total_moves = 0

    for x in range(map_size):
        for y in range(map_size):
            # Get best movement 
            model_best_directions = get_best_directions(output, x, y, map_size)
            ground_truth_best_directions = get_best_directions(ground_truth, x, y, map_size)

            # if model's best directions are a subset of the ground truth
            if set(model_best_directions).issubset(set(ground_truth_best_directions)):
                correct_moves += 1

            total_moves += 1

    accuracy = (correct_moves / total_moves) * 100
    print(f"Path Planning Accuracy for sample {index}: {accuracy:.2f}%")
    return accuracy



if __name__ == "__main__":
    MAP_SIZE = 15
    config = {
        "train_input_dest_path": "trainx15_small.npz",
        "train_output_dest_path": "trainy15_small.npz",
        "val_input_dest_path": "valx15_small.npz",
        "val_output_dest_path": "valy15_small.npz",
        "batch_size": 50,
        "epochs": 20,
        "learning_rate": 0.001,
    }

    if len(sys.argv) < 2:
        print("Usage: python script.py [t|v|a]")
        sys.exit(1)

    mode = sys.argv[1]

    input_data_path = config["train_input_dest_path"]
    output_data_path = config["train_output_dest_path"]

    if mode == "t":
        print("Starting training...")
        torch.autograd.set_detect_anomaly(True)
        train_data = SimpleDataset(x_file=config["train_input_dest_path"], y_file=config["train_output_dest_path"], train=True, batch_size=config["batch_size"])
        val_data = SimpleDataset(x_file=config["val_input_dest_path"], y_file=config["val_output_dest_path"], train=False, batch_size=config["batch_size"])

        train_loader = train_data.get_dataloader()
        val_loader = val_data.get_dataloader()

        model = WrappedModel(learning_rate=config["learning_rate"])
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",  # Monitor validation loss
            save_top_k=1,
            mode="min",
            dirpath="./checkpoints",
            filename="model-{epoch:02d}-{val_loss:.2f}"
        )
        trainer = pl.Trainer(
            max_epochs=config["epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[checkpoint_callback, LearningRateMonitor()],
            gradient_clip_val=1.0,  
            log_every_n_steps=1 
        )
        
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print("Training complete. Model saved in ./checkpoints.")
    
    elif mode == "v":
        print("Starting visualization...")
        # Automatically find the latest checkpoint file
        checkpoint_files = glob.glob("./checkpoints/*.ckpt")
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint files found in './checkpoints'. Ensure training has completed successfully.")
        
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)  # Get most recent checkpoint by modification time
        print(f"Using latest checkpoint: {latest_checkpoint}")

        visualize_output(
            model_checkpoint=latest_checkpoint,
            input_data_path=input_data_path,
            output_data_path=output_data_path,
            index=0,
            map_size=15
        )
    elif mode == "a":
        checkpoint_files = glob.glob("./checkpoints/*.ckpt")
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        compute_map_accuracy(
            model_checkpoint=latest_checkpoint,
            input_data_path=input_data_path,
            output_data_path=output_data_path,
            index=0,
            map_size=15
        )

        compute_accuracy_path(
            model_checkpoint=latest_checkpoint,
            input_data_path=input_data_path,
            output_data_path=output_data_path,
            index=0,
            map_size=15
        )

    else:
        print("Invalid mode. Use 't' for training, 'v' for visualization, and 'a' for accuracy.")
        sys.exit(1)