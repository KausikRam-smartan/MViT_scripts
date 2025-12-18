import time
import cv2
import os
import torch
import torch.nn as nn
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from ultralytics import YOLO
import numpy as np
import gc

from config import *


# Clear GPU memory
torch.cuda.empty_cache()
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'

# -----------------------------
# CONFIGURATION
# -----------------------------
RTSP_URL = "rtsp://admin:admin%40123@192.168.0.110:554/stream1"
BATCH_SIZE = 64
TARGET_FPS = 25
OUTPUT_FOLDER = '/home/smartan5070/Downloads/SlowfastTrainer-main/unseen_test/cropped_frames'
TRAIN_ROOT = train_datapath
MODEL_PATH = "/home/smartan5070/Downloads/SlowfastTrainer-main/downloaded_models/31_class_model_acc80_17_12.pt"
NUM_CLASSES = 31
K = 5


def clamp_bbox(bbox, h, w):
    """Clamp bounding box coordinates to ensure they are within image boundaries."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def load_model_from_local(model_path, num_classes, K, device):
    """Load model from local .pt file"""
    try:
        print(f"Loading model from local file: {model_path}")

        weights = MViT_V1_B_Weights.DEFAULT
        model = mvit_v1_b(weights=weights)

        for param in model.parameters():
            param.requires_grad = False

        last_fc_layer = model.head[-1]
        in_features = last_fc_layer.in_features
        # model.head[-1] = nn.Linear(in_features, num_classes)
        model.head[-1] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

        blocks = list(model.blocks)
        for block in blocks[-K:]:
            for p in block.parameters():
                p.requires_grad = True

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        print("✓ Model loaded successfully from local file!")
        return model

    except Exception as e:
        print(f"Failed to load from local file: {e}")
        return None


def run_inference_on_frames(frames, model, device, train_root):
    """Run inference directly on cropped frames without saving to video"""
    try:
        # Convert frames to tensor (frames are BGR from OpenCV)
        # Expected input: (C, T, H, W) where C=3, T=16, H=224, W=224

        # Resize all frames to 224x224
        resized = [cv2.resize(f, (224, 224)) for f in frames]

        # Sample 16 frames uniformly from the batch
        num_frames = len(resized)
        if num_frames >= 16:
            indices = np.linspace(0, num_frames - 1, 16).astype(int)
        else:
            # If less than 16, repeat last frame
            indices = list(range(num_frames)) + [num_frames - 1] * (16 - num_frames)

        sampled_frames = [resized[i] for i in indices]

        # Convert BGR to RGB and normalize
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled_frames]

        # Stack frames: (T, H, W, C)
        video_array = np.stack(rgb_frames, axis=0)

        # Convert to tensor and permute to (C, T, H, W)
        video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).float()

        # Normalize to [0, 1]
        video_tensor = video_tensor / 255.0

        # Apply normalization (ImageNet stats)
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        video_tensor = (video_tensor - mean) / std

        # Add batch dimension: (1, C, T, H, W)
        video_tensor = video_tensor.unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(video_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)

            # Convert to lists
            top3_probs = top3_probs[0].cpu().numpy()
            top3_indices = top3_indices[0].cpu().numpy()

        # Get class names
        class_names = sorted(os.listdir(train_root))

        # Build top 3 predictions
        top3_predictions = []
        for idx, prob in zip(top3_indices, top3_probs):
            class_name = class_names[idx] if idx < len(class_names) else f"UNKNOWN_{idx}"
            top3_predictions.append((class_name, prob))

        # Clean up
        del video_tensor, outputs
        torch.cuda.empty_cache()

        return top3_predictions

    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        return None


def clear_memory():
    """Explicitly clear memory"""
    gc.collect()
    torch.cuda.empty_cache()


def main():
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================
    # LOAD MVIT MODEL
    # ============================
    print("\n" + "="*60)
    print("LOADING MVIT MODEL")
    print("="*60)

    mvit_model = load_model_from_local(MODEL_PATH, NUM_CLASSES, K, device)

    if mvit_model is None:
        print("\n✗ FAILED TO LOAD MODEL!")
        exit(1)

    print("\n✓ MViT Model loaded and ready!")
    print("="*60 + "\n")

    # ============================
    # LOAD YOLO MODEL
    # ============================
    print("Loading YOLO model...")
    model_yolo = YOLO("yolov8n.pt").to(device)
    print("✓ YOLO model loaded!\n")

    # ============================
    # CONNECT TO RTSP STREAM
    # ============================
    print("Connecting to RTSP stream...")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("❌ Could not open RTSP stream")
        exit(1)

    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("❌ Cannot read frames from RTSP stream")
        cap.release()
        exit(1)

    print("✅ RTSP stream connected!")
    print(f"   Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}\n")

    # Variables for frame collection and processing
    cropped_buffer = []  # Buffer storing cropped person frames
    current_bbox = None
    current_top3_predictions = None  # Store top 3 predictions with probabilities
    prediction_history = []  # Store last 3 predictions for voting
    final_prediction = "Collecting frames..."

    print(f"Starting (Target: {BATCH_SIZE} cropped frames)")
    print("Press 'q' to quit\n")

    # Window name
    window_name = 'Action Recognition'

    print("Creating display window...")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  # Set larger window size
    print("✓ Window created\n")

    try:
        print("Entering main loop...")
        frame_loop_count = 0

        while True:
            ret, frame = cap.read()
            frame_loop_count += 1

            if frame_loop_count == 1:
                print("✓ First frame read successfully")

            if not ret or frame is None:
                print("⚠️ Reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(RTSP_URL)
                continue

            # Create display frame
            display_frame = frame.copy()

            # Detect person and crop immediately
            if len(cropped_buffer) < BATCH_SIZE:
                results = model_yolo(frame, conf=0.5, verbose=False)
                r = results[0]
                best_box = None
                max_area = 0
                h, w = frame.shape[:2]

                for box in r.boxes:
                    if int(box.cls.item()) == 0:  # person class
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        clamped = clamp_bbox((x1, y1, x2, y2), h, w)
                        if not clamped:
                            continue
                        x1, y1, x2, y2 = clamped
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            best_box = (int(x1), int(y1), int(x2), int(y2))

                if best_box is not None:
                    x1, y1, x2, y2 = best_box
                    PADDING = 15
                    x1 = max(0, x1 - PADDING)
                    y1 = max(0, y1 - PADDING)
                    x2 = min(w, x2 + PADDING)
                    y2 = min(h, y2 + PADDING)

                    # Crop and store
                    cropped = frame[y1:y2, x1:x2].copy()
                    cropped_buffer.append(cropped)
                    current_bbox = (x1, y1, x2, y2)

                    print(f"✓ Cropped frames: {len(cropped_buffer)}/{BATCH_SIZE}", end='\r')

                del results, r
                torch.cuda.empty_cache()

            # Draw bounding box and prediction if available
            if current_bbox is not None:
                x1, y1, x2, y2 = current_bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Draw top 3 predictions above bounding box
                if current_top3_predictions is not None:
                    text_y = max(y1 - 10, 100)
                    for i, (class_name, prob) in enumerate(current_top3_predictions):
                        prediction_text = f"{i+1}. {class_name}: {prob*100:.1f}%"
                        y_pos = text_y - (i * 30)

                        # Different colors for top 3
                        if i == 0:
                            color = (0, 255, 0)  # Green for top prediction
                            font_size = 1.0
                        elif i == 1:
                            color = (0, 255, 255)  # Yellow for 2nd
                            font_size = 0.8
                        else:
                            color = (0, 165, 255)  # Orange for 3rd
                            font_size = 0.7

                        cv2.putText(display_frame, prediction_text, (x1, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)
                else:
                    # Fall back to simple text if no predictions yet
                    text_y = max(y1 - 10, 30)
                    cv2.putText(display_frame, final_prediction, (x1, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Draw frame counter
            progress_text = f"Cropped: {len(cropped_buffer)}/{BATCH_SIZE}"
            cv2.putText(display_frame, progress_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show the frame
            cv2.imshow(window_name, display_frame)

            # Process when buffer is full
            if len(cropped_buffer) >= BATCH_SIZE:
                print(f"\n\nProcessing {len(cropped_buffer)} cropped frames...")

                try:
                    # Run MViT inference on cropped frames
                    print("Running action recognition inference...")
                    top3_predictions = run_inference_on_frames(cropped_buffer, mvit_model, device, TRAIN_ROOT)

                    if top3_predictions:
                        # Store the top 3 predictions for display
                        current_top3_predictions = top3_predictions

                        # Get the top prediction (highest probability)
                        top_class = top3_predictions[0][0]

                        # Add to prediction history
                        prediction_history.append(top_class)

                        # Keep only last 3 predictions
                        if len(prediction_history) > 3:
                            prediction_history.pop(0)

                        # Calculate mode (most common prediction)
                        from collections import Counter
                        vote_counts = Counter(prediction_history)
                        final_prediction = vote_counts.most_common(1)[0][0]

                        print(f"\nTop 3 predictions:")
                        for i, (class_name, prob) in enumerate(top3_predictions, 1):
                            print(f"  {i}. {class_name}: {prob*100:.2f}%")

                        print(f"\nHistory: {prediction_history}")
                        print(f"*** FINAL PREDICTION (mode of {len(prediction_history)} batches): {final_prediction} ***\n")
                    else:
                        print("✗ Inference failed\n")

                except Exception as e:
                    print(f"✗ Error: {e}")
                    import traceback
                    traceback.print_exc()

                finally:
                    # Clear buffer
                    cropped_buffer.clear()
                    clear_memory()
                    print("Ready for next batch\n")

            # Check for quit key (must be after imshow)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        clear_memory()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
