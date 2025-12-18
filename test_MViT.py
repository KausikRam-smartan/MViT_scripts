from decord import VideoReader, cpu
from config import *
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights

import mlflow
import mlflow.pytorch
import argparse
import os, time
import matplotlib.pyplot as plt
import seaborn as sns

from mlflow.tracking import MlflowClient


# Check your exact num_classes.
# num_classes = 31 
frames_per_clip = 16

# Specifying the model path
model_path = "/home/smartan5070/Downloads/SlowfastTrainer-main/Models/Trial_31class_11_12_25.pt"

# Defining the test datapath
test_datapath = "/home/smartan5070/Downloads/SlowfastTrainer-main/unseen_test_set_data/correct_videos 1/cropped_correct_videos"


# ------------------------------------------------------
# LOAD TRAINING CLASS ORDER (SAME AS TRAINING)
# ------------------------------------------------------
# train_root = "/home/smartan5070/Downloads/SlowfastTrainer-main/31_class_data_11_12_25_script_balanced_60_15/train"
train_root = train_datapath

# Sorted alphabetically → same as training
train_class_names = sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])

# Mapping from class name → index (same as training)
train_class_to_idx = {name: i for i, name in enumerate(train_class_names)}

# Reverse mapping index → class name
idx_to_class = {i: name for i, name in enumerate(train_class_names)}


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
        print("########### BUILD INDEX TRACKING (TRAINING ORDER) ###########")

        # ONLY accept folders that exist in training
        classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and d in train_class_to_idx
        ])

        print(f" |Test classes: {classes}")
        print(f" |Using training mapping: {train_class_to_idx}")

        self.class_to_idx = train_class_to_idx  # <- CRUCIAL

        for cls_name in classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(".mp4"):
                    self.video_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls_name])


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

            return frames, label

        except Exception as e:
            print(f"Failed to load video: {path}\nError: {e}")
            # try next video (avoid infinite recursion if dataset has 0 length)
            return self.__getitem__((idx + 1) % len(self))
        


def load_model(model_path, num_classes, K):
    
    weights = MViT_V1_B_Weights.DEFAULT
    
    # Load the model directly
    model = mvit_v1_b(weights=weights)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    last_fc_layer = model.head[-1]
    in_features = last_fc_layer.in_features
    # model.head[-1] = nn.Linear(in_features, num_classes)
    model.head[-1] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    #  Unfreeze the last K blocks (Crucial step! Must match training setup)
    blocks = list(model.blocks)
    for block in blocks[-K:]:
        for p in block.parameters():
            p.requires_grad = True

    # # 5. Load the state dictionary
    # # Use map_location to ensure it loads correctly regardless of the saved device
    # model_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    # Load the state dictionary
    # model_state_dict = torch.load(model_path)
    # Load the state dictionary INTO the instantiated model object
    # model.load_state_dict(model_state_dict)

    model = mlflow.pytorch.load_model(model_uri)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


def pack_pathway_output(frames, alpha=4):
    """
    Create inputs for SlowFast model from a clip.

    Args:
        frames (Tensor): shape (T, C, H, W)
        alpha (int): temporal stride between fast and slow pathways (usually 4)

    Returns:
        List[Tensor]: [slow_pathway, fast_pathway]
    """
    fast_pathway = frames # full frame sequence
    # print(f"29. Fast Pathway Shape: {fast_pathway.shape}")
    slow_indices = torch.linspace(0, fast_pathway.shape[2] - 1, fast_pathway.shape[2] // 4).long().to(device)
    # print(f"30. Line Spaced Slow Indices Using ({0, fast_pathway.shape[2] - 1, fast_pathway.shape[2] // 4})")
    slow_pathway = torch.index_select(fast_pathway, 2, slow_indices)  # T dimension is dim=2
    # print(f"31. Slow Pathway Shape: {slow_pathway.shape}")

    # print(f"Returning Pathways: [{slow_pathway.shape, fast_pathway.shape}]")
    return [slow_pathway, fast_pathway]

def run_infernce(test_loader):
    # Lists to store all true labels and predictions
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Inference'):
            inputs = inputs.to(device) 
            labels = labels.to(device)

            # Perform the forward pass
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            # Store the ground truth labels and predictions for metric calculation later
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return all_labels, all_predictions

def plot_confusion_matrix(conf_mat, class_names, save_path):
    plt.figure(figsize=(14, 12))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()



# # Main
# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--run_id", type=str, required=True,
#                         help="MLflow run ID to load the trained model")
#     args = parser.parse_args()

#     run_id = args.run_id
#     print(f"Using MLflow Run ID: {run_id}")

    
#     # 1️⃣ Set experiment FIRST
#     mlflow.set_experiment("MViT_Testing")

#     # 2️⃣ Create a NEW run for inference
#     mlflow.start_run(run_name=f"{run_id}_inference", nested=True)

#     # mlflow.start_run(run_name=run_id)
#     # mlflow.start_run(run_name=f"{run_id}_inference", nested=False)
#     mlflow.start_run(run_name=f"MViT-Finetune-{int(time.time())}")

#     model_uri = f"runs:/{run_id}/best_model"
#     print(f"Loading model from MLflow: {model_uri}")
    
#     model = load_model(model_uri, num_classes, K)

#     # Applying transform
#     transform = Compose([
#         Resize((256, 256)),
#         CenterCrop(224),
#         NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
#     ])

#     test_dataset = YourVideoDataset(test_datapath, transform=transform, frames_per_clip=frames_per_clip)

#     test_loader   = DataLoader(test_dataset,   batch_size=4, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda'))

#     all_labels, all_predictions = run_infernce(test_loader)

#     # Calculate Overall Accuracy
#     accuracy = accuracy_score(all_labels, all_predictions)
#     print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")

#     # Generate a detailed Classification Report (Precision, Recall, F1-Score)
#     # Determine which class indices actually appear in test data
#     present_classes = sorted(set(all_labels) | set(all_predictions))
#     # Map index back to class names for readability if you have a class_to_idx map
#     # target_names = list(test_dataset.class_to_idx.keys())
#     # Map these indices to class names
#     target_names = [idx_to_class[i] for i in present_classes]
#     print("\nClassification Report:")
#     print(classification_report(all_labels, all_predictions,labels=present_classes, target_names=target_names))

#     # Generate and print a Confusion Matrix
#     print("\nConfusion Matrix:")
#     conf_mat = confusion_matrix(all_labels, all_predictions)
#     # Prepare confusion matrix as plain text
#     conf_mat_text = "\nConfusion Matrix:\n" + np.array2string(conf_mat, separator=' ')
#     print(conf_mat)

#     # Log metrics
#     mlflow.log_metric("test_accuracy", accuracy)

#     # Log classification report as text
#     mlflow.log_text(classification_report(all_labels, all_predictions, target_names=target_names),
#                     artifact_file="test_classification_report.txt")
#     mlflow.log_text(conf_mat_text, "confusion_matrix.txt")

#     mlflow.end_run()


# Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True,
                        help="MLflow run ID to load the trained model")
    args = parser.parse_args()

    run_id = args.run_id
    print(f"Using MLflow Run ID: {run_id}")

    client = MlflowClient()

    # Fetch training run
    training_run = client.get_run(run_id)

    # MLflow stores description here
    training_description = training_run.data.tags.get(
        "mlflow.note.content", ""
    )

    print("Training run description:")
    print(training_description)
    
    # 1️⃣ Set experiment FIRST
    mlflow.set_experiment("MViT_Testing")

    # 2️⃣ Create ONE clean inference run
    with mlflow.start_run(run_name=f"MViT_{run_id}_inference"):

        # Copy training description into testing run
        if training_description:
            mlflow.set_tag("mlflow.note.content", training_description)

        model_uri = f"runs:/{run_id}/best_model"
        print(f"Loading model from MLflow: {model_uri}")

        model = load_model(model_uri, num_classes, K)

        # Applying transform
        transform = Compose([
            Resize((256, 256)),
            CenterCrop(224),
            NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])

        test_dataset = YourVideoDataset(test_datapath, transform=transform, frames_per_clip=frames_per_clip)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda'))

        all_labels, all_predictions = run_infernce(test_loader)

        # Accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")

        # Classification report
        present_classes = sorted(set(all_labels) | set(all_predictions))
        target_names = [idx_to_class[i] for i in present_classes]

        report = classification_report(all_labels, all_predictions,
                                    labels=present_classes,
                                    target_names=target_names)

        print("\nClassification Report:")
        print(report)

        conf_mat = confusion_matrix(
            all_labels,
            all_predictions,
            labels=list(idx_to_class.keys()) 
        )
        print("\nConfusion Matrix:")
        print(conf_mat)

        # Plot + save confusion matrix image
        cm_path = "confusion_matrix.png"
        plot_confusion_matrix(conf_mat, list(idx_to_class.values()), cm_path)

        # Log metrics + artifacts
        mlflow.log_param("training_run_id", args.run_id)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_text(report, "test_classification_report.txt")
        mlflow.log_text(np.array2string(conf_mat), "confusion_matrix.txt")

        # Log the confusion matrix image
        mlflow.log_artifact(cm_path)

