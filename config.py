import torch, os
import torch.nn as nn

# =========================
# Config
# =========================
# Train Settings
num_classes     = 9
num_epochs      = 12
learning_rate   = 1e-4
# optimizer_name  = "Adam"
optimizer_name  = "AdamW"
criterion       = nn.CrossEntropyLoss()
frames_per_clip = 32
K =5
dropout = 0.7
weight_decay = 0.05

# Paths
current_directory   = "/home/smartan5070/Downloads/SlowfastTrainer-main/Models/Testing_21Classes_Cam10718"
# model_save_path     = os.path.join(current_directory, "Testing_30Classes_Cam10718.pt")
model_save_path     = os.path.join(current_directory, "Trial_21class_12_12_25.pt")
# model_save_arch_path= os.path.join(current_directory, "architecture.pt")
model_save_arch_path= os.path.join(current_directory, "Trial_21class_12_12_25_architecture_21.pt")
# log_path            = os.path.join(current_directory, "SlowFast_training_log.txt")
log_path            = os.path.join(current_directory, "Trial_21class_12_12_25_log.txt")


# Data
train_datapath = "datasets/heirarchical_dataset_SET_217_cls/train"
val_datapath   = "datasets/heirarchical_dataset_SET_217_cls/val"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =========================
# Transforms (video-aware)
# =========================
import random
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class RandomRotateVideo:
    """
    Randomly rotates a video tensor shaped (C, T, H, W)
    """
    def __init__(self, degrees=10, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, video):
        # video: (C,T,H,W)
        if random.random() > self.p:
            return video

        angle = random.uniform(-self.degrees, self.degrees)

        C, T, H, W = video.shape
        frames = []

        for t in range(T):
            frame = video[:, t, :, :]  # (C,H,W)
            frame = F.rotate(
                frame,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False
            )
            frames.append(frame)

        return torch.stack(frames, dim=1)  # back to (C,T,H,W)
    
# class RandomHorizontalFlipVideo():
#     def __init__(self, p=0.5):
#         self.p = p
#         print('Horizontal flip inside class')
        
#     def __call__(self, video):
#         if random.random() > self.p:
#             return video
        
#         return torch.flip(video, dims=3)

# import random
# import torch
# import torchvision.transforms.functional as F
# from torchvision.transforms import InterpolationMode

class RandomShearVideo:
    """
    Randomly applies shear to a video tensor shaped (C, T, H, W)
    """
    def __init__(self, shear_degrees=10, p=0.5):
        """
        shear_degrees: max shear in degrees (same convention as torchvision affine)
        p: probability of applying the transform
        """
        self.shear_degrees = shear_degrees
        self.p = p

    def __call__(self, video):
        # video: (C, T, H, W)
        if random.random() > self.p:
            return video

        # shear can be (shear_x, shear_y)
        shear_x = random.uniform(-self.shear_degrees, self.shear_degrees)
        shear_y = random.uniform(-self.shear_degrees, self.shear_degrees)

        C, T, H, W = video.shape
        frames = []

        for t in range(T):
            frame = video[:, t, :, :]  # (C, H, W)

            frame = F.affine(
                frame,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[shear_x, shear_y],
                interpolation=InterpolationMode.BILINEAR
            )

            frames.append(frame)

        return torch.stack(frames, dim=1)  # back to (C, T, H, W)


try:
    print("_transforms_video is available")
    from torchvision.transforms._transforms_video import ResizeVideo, NormalizeVideo, RandomHorizontalFlip
    from torchvision.transforms import Compose
    transform = Compose([
        ResizeVideo((224, 224)),                               # (C,T,H,W)
        RandomHorizontalFlip(p=0.9),
        RandomRotateVideo(degrees=15, p=0.9),
        RandomShearVideo(shear_degrees=20, p=0.9),
        NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])
except Exception:
    print("Fallback _transforms_video not available")
    # Fallback if _transforms_video not available (kept for compatibility)
    from torchvision.transforms import Compose, Resize
    from torchvision.transforms._transforms_video import NormalizeVideo, RandomHorizontalFlipVideo
    transform = Compose([
        Resize((224, 224)),                                    # Works for some torchvision versions on (C,T,H,W)
        RandomHorizontalFlipVideo(p=0.9),
        RandomRotateVideo(degrees=15, p=0.9),
        RandomShearVideo(shear_degrees=15, p=0.9),
        NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])


