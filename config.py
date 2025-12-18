import torch, os
import torch.nn as nn

# =========================
# Config
# =========================
# Train Settings
num_classes     = 31
num_epochs      = 10
learning_rate   = 1e-4
# optimizer_name  = "Adam"
optimizer_name  = "AdamW"
criterion       = nn.CrossEntropyLoss()
frames_per_clip = 32
K = 5

# Paths
current_directory   = "/home/smartan5070/Downloads/SlowfastTrainer-main/Models/Testing_21Classes_Cam10718"
# model_save_path     = os.path.join(current_directory, "Testing_30Classes_Cam10718.pt")
model_save_path     = os.path.join(current_directory, "Trial_21class_12_12_25.pt")
# model_save_arch_path= os.path.join(current_directory, "architecture.pt")
model_save_arch_path= os.path.join(current_directory, "Trial_21class_12_12_25_architecture_21.pt")
# log_path            = os.path.join(current_directory, "SlowFast_training_log.txt")
log_path            = os.path.join(current_directory, "Trial_21class_12_12_25_log.txt")


# Data
train_datapath = "/home/smartan5070/Downloads/SlowfastTrainer-main/dataset_31_class/31_class_16_12_25_balanced_script_70_15/train"
val_datapath   = "/home/smartan5070/Downloads/SlowfastTrainer-main/dataset_31_class/31_class_16_12_25_balanced_script_70_15/val"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =========================
# Transforms (video-aware)
# =========================
try:
    print("_transforms_video is available")
    from torchvision.transforms._transforms_video import ResizeVideo, NormalizeVideo, RandomHorizontalFlip
    from torchvision.transforms import Compose
    transform = Compose([
        ResizeVideo((224, 224)),                               # (C,T,H,W)
        RandomHorizontalFlip(p=0.5),
        NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])
except Exception:
    print("Fallback _transforms_video not available")
    # Fallback if _transforms_video not available (kept for compatibility)
    from torchvision.transforms import Compose, Resize
    from torchvision.transforms._transforms_video import NormalizeVideo, RandomHorizontalFlipVideo
    transform = Compose([
        Resize((224, 224)),                                    # Works for some torchvision versions on (C,T,H,W)
        RandomHorizontalFlipVideo(p=0.5),
        NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])
