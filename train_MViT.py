from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Lambda
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader

from config import *
import os, time
from tqdm import tqdm
import numpy as np
import shutil
import argparse

import mlflow
import mlflow.pytorch

# Move the model to the appropriate device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YourVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=None):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.video_paths = []
        self.labels = []
        self.class_to_idx = {}
        self._build_index()

    def _build_index(self):
        print("########### BUILD INDEX TRACKING ###########")
        classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        print(f" |Classes: {classes}")
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f" |Class_to_idx: {self.class_to_idx}")
        for cls_name in classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            print(f" |Class_directory: {cls_dir}")
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(".mp4"):
                    self.video_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls_name])
        print(f" |Num videos: {len(self.video_paths)}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        try:
            vr = VideoReader(path, ctx=cpu(0))
            total_frames = len(vr)

            # Frame indices
            if total_frames < self.frames_per_clip:
                # pad by repeating last frame (simple + robust)
                base = np.linspace(0, total_frames - 1, total_frames).astype(int)
                pad = self.frames_per_clip - total_frames
                frame_indices = np.concatenate([base, np.full((pad,), base[-1], dtype=int)])
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip).astype(int)

            frames = vr.get_batch(frame_indices).asnumpy()          # (T,H,W,C)

            # Fix for grayscale videos
            if frames.shape[-1] == 1:
                frames = np.repeat(frames, 3, axis=-1)
            elif frames.shape[-1] != 3:
                raise ValueError(f"Unsupported channel count: {frames.shape[-1]} in video {path}")

            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0   # (C,T,H,W)

            if self.transform:
                frames = self.transform(frames)                                    # keep (C,T,H,W)

            # # return (T,C,H,W) to match your later permute call
            # frames = frames.permute(1, 0, 2, 3).contiguous()                       # (T,C,H,W)

            return frames, label

        except Exception as e:
            print(f"Failed to load video: {path}\nError: {e}")
            # try next video (avoid infinite recursion if dataset has 0 length)
            return self.__getitem__((idx + 1) % len(self))
        

# =========================
# Load the Pretrained Model
# =========================

def load_pretrained_model():
    # 1. Define the desired weights
    weights = MViT_V1_B_Weights.DEFAULT  # or KINETICS400_V1 depending on your task
    # weights = MViT_V2_S_Weights.DEFAULT  # Use MViT V2 weights


    # 2. Load the model with the pretrained weights
    model = mvit_v1_b(weights=weights)
    # model = mvit_v2_s(weights=weights)

    # =========================
    # Freezing Layers and Modifying the Head
    # =========================

    # --- FREEZE ALL LAYERS IN THE BACKBONE ---
    for param in model.parameters():
        param.requires_grad = False

    # The last layer in the `head` will be a Linear layer, so we access it as follows:
    last_fc_layer = model.head[-1]  # Get the last layer of the head (should be a Linear layer)
    in_features = last_fc_layer.in_features  # Get the number of input features to this layer

    # Replace the last Linear layer in `head` with a new one that has the number of output classes you need
    # num_classes = 21  # Replace with your own number of classes
    # model.head[-1] = nn.Linear(in_features, num_classes)
    model.head[-1] = nn.Sequential(
        nn.Dropout(p=0.8),
        nn.Linear(in_features, num_classes)
    )

    print(f"Model ready for fine-tuning with {num_classes} output classes.")

    # Unfreezing few layers(Total 12 layers for MVit model)
    # K =  2
    blocks = list(model.blocks)

    for block in blocks[-K:]:
        for p in block.parameters():
            p.requires_grad =True

    # =========================
    # Optimizer
    # =========================

    optimizer_name = "Adam"  # This should be defined before or passed as an argument
    learning_rate = 1e-4  # Set learning rate as per your requirement



    if optimizer_name == "Adam":
        print(f"Activating Optimizer {optimizer_name}")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # mlflow.log_param("K", K)

    return model,optimizer

# =========================
# Training Loop
# =========================

def overwrite_mlflow_artifact(artifact_name):
    artifact_path = mlflow.get_artifact_uri(artifact_name)
    if artifact_path.startswith("file://"):
        artifact_path = artifact_path.replace("file://", "")
    if os.path.exists(artifact_path):
        shutil.rmtree(artifact_path)

def train_model(model,optimizer):

    model = model.to(device)

    # Loss function (cross entropy for classification)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # num_epochs = 10

    best_loss = float('inf')
    start_time = time.time()



    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # Set the model in training mode
        model.train()
        
        # Initialize variables to track metrics
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        all_train_preds = []
        all_train_labels = []

        # Iterate through the training dataset
        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move inputs and targets to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients for each batch
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, targets)
            
            # Backward pass (compute gradients)
            loss.backward()

            # Check gradient norms to avoid zero/NaN gradients
            any_finite = False
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    g = p.grad.detach()
                    gnorm = g.data.norm(2).item()
                    if np.isfinite(gnorm) and gnorm > 0:
                        any_finite = True

            if not any_finite:
                print("[grad check] All parameter grad norms are zero/NaN this step.")

            # Compute total gradient norm (to monitor exploding/vanishing gradients)
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    pn = p.grad.data.norm(2).item()
                    total_norm_sq += pn * pn
            total_grad_norm = float(total_norm_sq ** 0.5)
            
            # Update model parameters
            optimizer.step()
            
            # Track statistics for loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            # Collect epoch training metrics
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_train_preds.extend(preds.tolist())
            all_train_labels.extend(targets.detach().cpu().numpy().tolist())
            
        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples * 100

        # Epoch metrics
        avg_train_loss = running_loss / max(1, len(train_loader))
        train_accuracy = accuracy_score(all_train_labels, all_train_preds) if all_train_labels else 0.0
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0) if all_train_labels else 0.0
        train_recall    = recall_score(all_train_labels, all_train_preds,   average='weighted', zero_division=0) if all_train_labels else 0.0
        train_f1        = f1_score(all_train_labels, all_train_preds,       average='weighted', zero_division=0) if all_train_labels else 0.0

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Training Loss: {epoch_loss:.4f}")
        print(f"  Training Accuracy: {epoch_accuracy:.2f}%")

        # =========================
        # Validation Phase
        # =========================
        
        # Set the model in evaluation mode
        model.eval()

        # Disable gradient calculation for validation (saves memory)
        with torch.no_grad():
            val_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            all_val_preds = [] 
            all_val_targets = []
            
            # Iterate through the validation dataset
            for inputs, targets in val_loader:
                # Move inputs and targets to the correct device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                
                # Compute the loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                all_val_preds.extend(predicted.detach().cpu().numpy())
                all_val_targets.extend(targets.detach().cpu().numpy())
            
            # Calculate average loss and accuracy for validation
            val_loss /= len(val_loader)
            val_accuracy = correct_predictions / total_samples * 100

            eval_accuracy = correct_predictions / total_samples if total_samples else 0.0
            eval_precision = precision_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
            eval_recall    = recall_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
            eval_f1        = f1_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
            
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.2f}%")

        # Log training & validation metrics to MLflow
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,

            "val_loss": val_loss,
            "val_accuracy": eval_accuracy,
            "val_precision": eval_precision,
            "val_recall": eval_recall,
            "val_f1": eval_f1,

            "grad_norm": total_grad_norm
        }, step=epoch)

        saved= False
        # Checkpoint Logic
        if val_loss < best_loss:
            print(f"  *** Validation Loss improved from {best_loss:.4f} to {val_loss:.4f}. Saving model. ***")
            best_loss = val_loss
            
            # Save only the model's parameters (state_dict)
            torch.save(model.state_dict(), model_save_path)
            torch.save(model, model_save_arch_path)

            
            saved= True
        else:
            print(f"  Validation Loss did not improve.")

        print("\n=== Logging final BEST model to MLflow ===")

        # Logs
        print(f"Train Loss: {avg_train_loss:.6f}, Acc: {train_accuracy:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Eval  Loss: {val_loss:.6f}, Acc: {eval_accuracy:.4f}, Prec: {eval_precision:.4f}, Rec: {eval_recall:.4f}, F1: {eval_f1:.4f}")
        print(f"Grad Norm (last step): {total_grad_norm:.6f}, LR: {optimizer.param_groups[0]['lr']:.10f}")
        if saved:
            print(f"Model saved to: {os.path.abspath(model_save_path)}")
        print(f"Epoch Time Taken: {(time.time() - epoch_start_time):.2f} sec")

        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}\n")
            f.write(f"Train Loss: {avg_train_loss:.6f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}\n")
            f.write(f"Eval Loss: {val_loss:.6f}, Accuracy: {eval_accuracy:.4f}, Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}, F1: {eval_f1:.4f}\n")
            f.write(f"Grad Norm: {total_grad_norm:.6f}, Learning Rate: {optimizer.param_groups[0]['lr']:.10f}\n")
            if saved:
                f.write(f"Model saved to: {os.path.abspath(model_save_path)}\n")
            f.write(f"Epoch Time Taken: {(time.time() - epoch_start_time):.2f} sec\n")
            f.write("----------------------------------------------------------\n")
    # Load the best model again
    best_model = load_pretrained_model()[0]   # Same architecture
    best_model.load_state_dict(torch.load(model_save_path))
    best_model = best_model.to(device)

    overwrite_mlflow_artifact("best_model")
    mlflow.pytorch.log_model(best_model, name="best_model")

    print("BEST model logged to MLflow.")

# Main
if __name__ == "__main__":

    # ======================
    # Initialize MLflow
    # ======================

    mlflow.set_experiment("VIDEO-ACTION-CLASSIFICATION")
    run = mlflow.start_run(run_name=f"MViT-Finetune-{int(time.time())}")
    run_id = run.info.run_id

    # Save run_id inside the experiment
    mlflow.log_param("run_id", run_id)

    # =========================
    # Datasets and DataLoader Setup
    # =========================

    train_dataset = YourVideoDataset(train_datapath, transform=transform, frames_per_clip=16)
    val_dataset = YourVideoDataset(val_datapath, transform=transform, frames_per_clip=16)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

    model,optimizer = load_pretrained_model()

    # Log hyperparameters
    mlflow.log_params({
        "epochs": num_epochs,
        "K" : K,
        "Num_classes": num_classes,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "batch_size": 4,
        "frames_per_clip": 16,
        "model": "MViT_V1_B",
        "train_videos": len(train_dataset),
        "val_videos": len(val_dataset)
    })


    def count_trainable_parameters(model):
        """ Counts the total number of trainable parameters in a PyTorch model. """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print number of trainable parameters
    total_trainable_params = count_trainable_parameters(model)
    print(f"\nTotal trainable parameters in the model: {total_trainable_params:,}")

    train_model(model,optimizer)

    mlflow.end_run()
