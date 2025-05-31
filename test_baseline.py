import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from finetune_baseline import VideoCSVDataset
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score, accuracy_score
from torchvision import transforms
from torchvision.models.video import mvit_v2_s, swin3d_s
from torchvision.models.video import MViT_V2_S_Weights, Swin3D_S_Weights


def evaluate_dataset(model, checkpoint_path, csv_file, video_root, transform, batch_size=8, device='cuda'):
    # Load model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
    print(f"Evaluating model from {checkpoint_path} on dataset {dataset_name}")
    
    test_dataset = VideoCSVDataset(
        csv_file=csv_file,
        root_dir=video_root,
        transform=transform,
        split='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            probs = F.softmax(outputs, dim=1)[:, 1]  # probability for class 1 (fake)
            preds = (probs > 0.5).long()  # threshold at 0.5

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Computing metrics for class 1 (fake)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    ap = average_precision_score(all_labels, all_probs, pos_label=1)
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"{dataset_name} -> Precision: {precision:.3f}, Recall: {recall:.3f}, AP: {ap:.3f}, Accuracy: {accuracy:.3f}")

    return {
        "dataset": dataset_name,
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "accuracy": accuracy
    }


def create_model(model_type):
    """Create and modify model for binary classification"""
    if model_type.lower() == 'mvit':
        model = mvit_v2_s(weights=None)
        transform = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    elif model_type.lower() == 'swin':
        model = swin3d_s(weights=None)
        transform = Swin3D_S_Weights.KINETICS400_V1.transforms()
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


def main():
    parser = argparse.ArgumentParser(description='Test baseline video classification model')
    
    parser.add_argument('--model', type=str, choices=['mvit', 'swin'], required=True, help='Model type to use (mvit or swin)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with test data')
    parser.add_argument('--video_root', type=str, default = 'video_data', required=True, help='Root directory containing videos')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    model, transform = create_model(args.model)

    results = evaluate_dataset(
        model=model,
        checkpoint_path=args.checkpoint,
        csv_file=args.csv,
        video_root=args.video_root,
        transform=transform,
        batch_size=args.batch_size,
        device=device
    )

    print("\nFinal Results:")
    print(f"Dataset: {results['dataset']}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"Average Precision: {results['ap']:.3f}")
    print(f"Accuracy: {results['accuracy']:.3f}")


if __name__ == "__main__":
    main()