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


def predict(
    image_input_folder,
    npy_output_folder,
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

    for fname in os.listdir(image_input_folder):
        if fname.lower().endswith(".jpg"):
            image_path = os.path.join(image_input_folder, fname)
            img, mask_pred = predict_image(model, image_path)

            npy_output_path = os.path.join(
                npy_output_folder, fname.lower().replace("jpg", "npy")
            )
            np.save(npy_output_path, mask_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-image_input_folder", type=str)
    parser.add_argument("-npy_output_folder", type=str)
    parser.add_argument("-n_classes", type=int)
    parser.add_argument("--train_folder_path", type=str)
    parser.add_argument("--best_epoch", type=int)
    parser.add_argument("--model_architecture", type=str, default="Unet")
    parser.add_argument("--encoder_name", type=str, default="resnet18")
    args = parser.parse_args()

    image_input_folder = args.image_input_folder
    npy_output_folder = args.npy_output_folder
    n_classes = args.n_classes
    train_folder_path = args.train_folder_path
    best_epoch = args.best_epoch
    model_architecture = args.model_architecture
    encoder_name = args.encoder_name

    os.makedirs(npy_output_folder, exist_ok=True)

    predict(
        image_input_folder,
        npy_output_folder,
        n_classes,
        train_folder_path,
        best_epoch,
        model_architecture,
        encoder_name,
    )
