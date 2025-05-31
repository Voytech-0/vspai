import os
import argparse
import pandas as pd
import torch
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import ToTensor
import decord
from decord import VideoReader, cpu
from PIL import Image
import torch.nn as nn


decord.bridge.set_bridge('torch')


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Baseline on video dataset")

    parser.add_argument('--csv_path', type=str, required = True, help='Path to CSV metadata')
    parser.add_argument('--root_dir', type=str, default='video_data', help='Root directory of video files')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model', type=str, choices=['mvit', 'swin'], required=True, help='Model type to use (mvit or swin)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')

    return parser.parse_args()


# === Frame and Dataset ===
decord.bridge.set_bridge('torch')

def get_frame(filepath: str, index: int):
    try:
        vr = decord.VideoReader(filepath, ctx=decord.cpu(0))
        image = vr[index].cpu().numpy() # different from spav usage
        image = Image.fromarray(image)
        del vr
    except Exception as e:
        print(f"Image not read properly: {filepath} | Exception: {e}")
        image = Image.new("RGB", (128, 128), (0, 0, 0))
    return image

def get_num_frames(filepath: str):
    try:
        vr = decord.VideoReader(filepath, ctx=decord.cpu(0))
        length = len(vr)
        del vr
        return length
    except Exception as e:
        print(f"Failed to read video length: {filepath} | Exception: {e}")
        return 0

class VideoCSVDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, num_frames=16, split='train'):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.split = split
        if 'split' in self.data.columns:
            self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        print(f"Dataset initialized with {len(self.data)} samples for split '{split}'.")

    def __len__(self):
        return len(self.data)

    def _load_video(self, video_path):
        total_frames = get_num_frames(video_path)
        # print(f"Loading video: {video_path} | Total frames: {total_frames}")

        if total_frames == 0:
            print(f"Warning: Video {video_path} has zero frames, returning zeros tensor.")
            return torch.zeros(self.num_frames, 3, 224, 224)

        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        else:
            # padding by repeating last frame index as some vids shorter than 16 frames
            indices = list(range(total_frames)) + [total_frames - 1] * (self.num_frames - total_frames)

        # print(f"Frame indices sampled: {indices}")

        frames_list = []
        for idx in indices:
            img = get_frame(video_path, idx)
            img = img.resize((224, 224))
            img_tensor = ToTensor()(img)  # (C, H, W), values scaled [0,1]
            frames_list.append(img_tensor)

        frames = torch.stack(frames_list)  # (T, C, H, W)
        return frames

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = os.path.join(self.root_dir, row['image'])
        label = int(row['class'])
        # print(f"Getting item {idx}: {video_path} with label {label}")

        try:
            frames = self._load_video(video_path)
        except Exception as e:
            print(f"Failed to load video {video_path}: {e}")
            frames = torch.zeros((self.num_frames, 3, 224, 224))

        if self.transform:
            frames = self.transform(frames)  # expects (T, C, H, W) or (B, T, C, H, W)
            # print(f"Applied transform to frames, resulting shape: {frames.shape}")

        return frames, label


def create_model(model_type):
    if model_type.lower() == 'mvit':
        model = mvit_v2_s(weights=None)
        transform = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    elif model_type.lower() == 'swin':
        model = swin3d_t(weights=None)
        transform = Swin3D_T_Weights.KINETICS400_V1.transforms()
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'mvit' or 'swin'")


    if isinstance(model.head, nn.Sequential):
        old_head = model.head
        last_linear = None
        for layer in reversed(old_head):
            if isinstance(layer, nn.Linear):
                last_linear = layer
                break

        if last_linear is None:
            raise RuntimeError("No linear layer found in model.head")

        in_features = last_linear.in_features
        new_head = list(old_head.children())[:-1] + [nn.Linear(in_features, 2)]
        model.head = nn.Sequential(*new_head)
    else:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 2)

    return model, transform

# === Main Training Logic ===
def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    model, transform = create_model(args.model)
    
    train_dataset = VideoCSVDataset(csv_file=args.csv_path, root_dir=args.root_dir, transform=transform, split='train')
    val_dataset = VideoCSVDataset(csv_file=args.csv_path, root_dir=args.root_dir, transform=transform, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if isinstance(model.head, nn.Sequential):
        last_linear = next((l for l in reversed(model.head) if isinstance(l, nn.Linear)), None)
        if last_linear is None:
            raise RuntimeError("No linear layer found in model head")
        in_features = last_linear.in_features
        new_head = list(model.head.children())[:-1] + [nn.Linear(in_features, 2)]
        model.head = nn.Sequential(*new_head)
    else:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 2)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)


        train_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # === Validation ===
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_epoch_loss = val_loss / len(val_loader)
        print(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.2f}%")


if __name__ == "__main__":
    main()
