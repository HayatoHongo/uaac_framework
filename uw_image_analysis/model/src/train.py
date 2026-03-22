import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import SegmentationDataset
from utils.save_checkpoint import save_checkpoint
import numpy as np
from time import time
import random
import os
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.7, device=0)

import albumentations as A

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved: {path}")


def compute_iou_multiclass(pred, target, num_classes, eps=1e-7):
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            continue
        ious.append((intersection + eps) / (union + eps))
    if len(ious) == 0:
        return np.nan
    return np.mean(ious)


def train(
    n_classes,
    max_epochs,
    batch_size,
    model_architecture,
    encoder_name,
    encoder_weights,
    patience,
    use_data_augmentation,
    use_data_augmentation_color_jitter,
    dataset_name,
    ce_weight,
    dice_weight,
    focal_weight,
    show_loss_log_plot,
    show_iou_log_plot,
):
    available_model_architectures = ["DeepLabV3", "Unet"]
    if model_architecture not in available_model_architectures:
        print(f"model architecture: {model_architecture} is not available.")
        return

    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_log_name = f"{dataset_name}_{now_str}"
    train_output_folder_path = os.path.join(
        "uw_image_analysis/model/outputs", train_log_name
    )

    checkpoint_folder = os.path.join(train_output_folder_path, "checkpoints")
    os.makedirs(checkpoint_folder)

    parameter_log_file_name = os.path.join(train_output_folder_path, "params.txt")
    loss_log_file_name = os.path.join(train_output_folder_path, "losses.txt")

    with open(parameter_log_file_name, "w", encoding="utf-8") as f:
        f.write(f"model: {model_architecture}\n")
        f.write(f"encoder name: {encoder_name}\n")
        f.write(f"encoder weights: {encoder_weights}\n")
        f.write(f"dataset name: {dataset_name}\n")
        f.write(f"n classes: {n_classes}\n")
        f.write(f"max epochs: {max_epochs}\n")
        f.write(f"patience: {patience}\n")
        f.write(f"batch size: {batch_size}\n")
        f.write(f"data augmentation: {str(use_data_augmentation)}\n")
        f.write(f"  color jitter: {str(use_data_augmentation_color_jitter)}\n")
        f.write(f"cross entropy weight: {str(ce_weight)}\n")
        f.write(f"dice weight: {str(dice_weight)}\n")
        f.write(f"focal weight: {str(focal_weight)}\n")

    train_transform = A.Compose(
        A.Compose(
            [
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )
    )

    val_transform = A.Compose([A.Resize(512, 512)])
    train_dataset = SegmentationDataset(
        f"uw_image_analysis/data/{dataset_name}/train/images",
        f"uw_image_analysis/data/{dataset_name}/train/masks",
        transform=train_transform if use_data_augmentation else val_transform,
    )
    val_dataset = SegmentationDataset(
        f"uw_image_analysis/data/{dataset_name}/test/images",
        f"uw_image_analysis/data/{dataset_name}/test/masks",
        transform=val_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    if model_architecture == "DeepLabV3":
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=n_classes,
        ).to(DEVICE)
    elif model_architecture == "Unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=n_classes,
        ).to(DEVICE)
    else:
        return

    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    test_losses = []
    test_ious = []

    best_iou = 0
    best_epoch = 0
    counter = 0

    for epoch in range(max_epochs):
        start_time = time()
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        end_time = time()

        train_losses.append(train_loss / len(train_loader))

        print(
            f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Time: {end_time-start_time}"
        )
        with open(loss_log_file_name, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Time: {end_time-start_time}\n"
            )

        model.eval()
        val_loss = 0
        iou_list = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                loss = loss_fn(preds, masks)
                val_loss += loss.item()

                pred_masks = torch.argmax(preds, dim=1)
                masks_sq = masks.squeeze(1)

                for p, t in zip(pred_masks, masks_sq):
                    iou = compute_iou_multiclass(
                        p.cpu(), t.cpu(), num_classes=n_classes
                    )
                    iou_list.append(iou)

        val_loss_epoch = val_loss / len(val_loader)
        test_losses.append(val_loss_epoch)
        mean_iou = np.nanmean(iou_list)
        test_ious.append(mean_iou)

        print(f"  Val Loss: {val_loss_epoch:.4f}, Mean IoU: {mean_iou:.4f}")
        with open(loss_log_file_name, "a", encoding="utf-8") as f:
            f.write(f"  Val Loss: {val_loss_epoch:.4f}, Mean IoU: {mean_iou:.4f}\n")

        if mean_iou > best_iou:
            best_iou = mean_iou
            best_epoch = epoch
            counter = 0
            print(f"  ✨ Mean IoU improved!")

            save_checkpoint(model, f"{checkpoint_folder}/epoch{epoch+1}.pth")
        else:
            counter += 1
            print(f"  ⚠️ No improvement for {counter} epochs")

            if counter >= patience:
                print(f"\n⏹ Early stopping triggered at epoch {epoch+1}")
                print(
                    f"Best model from epoch {best_epoch+1} with mean_iou={best_iou:.4f}"
                )
                with open(loss_log_file_name, "a", encoding="utf-8") as f:
                    f.write(
                        f"Best model from epoch {best_epoch+1} with mean_iou={best_iou:.4f}"
                    )
                break

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(test_losses, label="Test Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(train_output_folder_path, "loss_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show_loss_log_plot:
        plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(test_ious, label="Test Mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("iou")
    plt.title("iou history")
    plt.grid(True)
    plt.savefig(
        os.path.join(train_output_folder_path, "iou_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show_iou_log_plot:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_architecture", type=str, default="Unet")
    parser.add_argument("--encoder_name", type=str, default="resnet18")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("-dataset_name", type=str)
    parser.add_argument("-n_classes", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_data_augmentation", action="store_true")
    parser.add_argument("--use_data_augmentation_color_jitter", action="store_true")
    parser.add_argument("--ce_weight", type=float, default=1)
    parser.add_argument("--dice_weight", type=float, default=0)
    parser.add_argument("--focal_weight", type=float, default=0)
    parser.add_argument("--show_loss_log_plot", action="store_true")
    parser.add_argument("--show_iou_log_plot", action="store_true")

    args = parser.parse_args()

    model_architecture = args.model_architecture
    encoder_name = args.encoder_name
    encoder_weights = args.encoder_weights
    max_epochs = args.max_epochs
    patience = args.patience
    dataset_name = args.dataset_name
    n_classes = args.n_classes
    batch_size = args.batch_size
    use_data_augmentation = args.use_data_augmentation
    use_data_augmentation_color_jitter = args.use_data_augmentation_color_jitter
    ce_weight = args.ce_weight
    dice_weight = args.dice_weight
    focal_weight = args.focal_weight
    show_loss_log_plot = args.show_loss_log_plot
    show_iou_log_plot = args.show_iou_log_plot

    train(
        n_classes,
        max_epochs,
        batch_size,
        model_architecture,
        encoder_name,
        encoder_weights,
        patience,
        use_data_augmentation,
        use_data_augmentation_color_jitter,
        dataset_name,
        ce_weight,
        dice_weight,
        focal_weight,
        show_loss_log_plot,
        show_iou_log_plot,
    )
