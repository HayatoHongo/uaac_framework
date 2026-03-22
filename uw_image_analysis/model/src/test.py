import argparse
import os
import random
import numpy as np
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.7, device=0)
from sklearn.metrics import (
    confusion_matrix,
)

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


def load_model(ckpt_path, model_architecture, encoder_name, n_classes):
    available_model_architectures = ["DeepLabV3", "Unet"]

    if model_architecture not in available_model_architectures:
        print(f"model architecture: {model_architecture} is not available.")
        return

    if model_architecture == "DeepLabV3":
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=n_classes,
        ).to(DEVICE)
    elif model_architecture == "Unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=n_classes,
        ).to(DEVICE)
    else:
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model


def predict_image(model, img_path):
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    image = Image.open(img_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    return image, pred_mask


def test(
    dataset_name,
    n_classes,
    train_folder_path,
    best_epoch,
    model_architecture,
    encoder_name,
):
    model = load_model(
        f"uw_image_analysis/model/outputs/{train_folder_path}/checkpoints/epoch{str(best_epoch)}.pth",
        model_architecture,
        encoder_name,
        n_classes,
    )
    image_input_folder = f"uw_image_analysis/data/{dataset_name}/test/images"
    mask_input_folder = f"uw_image_analysis/data/{dataset_name}/test/masks"

    cm_total = np.zeros((n_classes, n_classes), dtype=int)

    for fname in os.listdir(image_input_folder):
        if fname.lower().endswith(".jpg"):
            image_path = os.path.join(image_input_folder, fname)
            img, mask_pred = predict_image(model, image_path)

            mask_path = (
                os.path.join(mask_input_folder, fname.replace("jpg", "png"))
                if mask_input_folder != ""
                else None
            )
            if mask_path is not None:
                mask_true = np.array(Image.open(mask_path))

                y_true = mask_true.flatten()
                y_pred = mask_pred.flatten()

                valid_mask = y_true != 255
                y_true = y_true[valid_mask]
                y_pred = y_pred[valid_mask]

                cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
                cm_total += cm

    TP = np.diag(cm_total)
    FP = cm_total.sum(axis=0) - TP
    FN = cm_total.sum(axis=1) - TP
    TN = cm_total.sum() - (TP + FP + FN)

    # Accuracy（micro）
    accuracy = TP.sum() / cm_total.sum()

    # Precision / Recall / F1
    precision_per_class = TP / np.maximum(TP + FP, 1)
    recall_per_class = TP / np.maximum(TP + FN, 1)
    f1_per_class = (
        2
        * precision_per_class
        * recall_per_class
        / np.maximum(precision_per_class + recall_per_class, 1e-8)
    )

    precision = np.nanmean(precision_per_class)
    recall = np.nanmean(recall_per_class)
    f1 = np.nanmean(f1_per_class)

    # IoU / mIoU
    iou_per_class = TP / np.maximum(TP + FP + FN, 1)
    miou = np.nanmean(iou_per_class)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)
    print("mIoU     :", miou)
    print("cm total")
    print(cm_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_name", type=str)
    parser.add_argument("-n_classes", type=int)
    parser.add_argument("--train_folder_path", type=str)
    parser.add_argument("--best_epoch", type=int)
    parser.add_argument("--model_architecture", type=str, default="Unet")
    parser.add_argument("--encoder_name", type=str, default="resnet18")
    parser.add_argument("--show_image", action="store_true")
    parser.add_argument("--show_each_result", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    n_classes = args.n_classes
    train_folder_path = args.train_folder_path
    best_epoch = args.best_epoch
    model_architecture = args.model_architecture
    encoder_name = args.encoder_name
    show_image = args.show_image
    show_each_result = args.show_each_result

    test(
        dataset_name,
        n_classes,
        train_folder_path,
        best_epoch,
        model_architecture,
        encoder_name,
        show_image,
        show_each_result,
    )
