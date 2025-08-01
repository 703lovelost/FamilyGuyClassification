#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from torchvision.models import resnet34, ResNet34_Weights
from torchvision.datasets import ImageFolder

import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_model(model_path: str, device: torch.device):
    """
    Загружает checkpoint, извлекает state_dict и список классов (если есть),
    восстанавливает модель и возвращает (model, classes).
    """
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        classes = ckpt.get("classes", None)
    else:
        state_dict = ckpt
        classes = None

    model = resnet34(weights=None)
    num_classes = len(classes) if classes is not None else state_dict["fc.weight"].shape[0]
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, classes

def predict(model: torch.nn.Module,
            classes: list[str],
            image_path: str,
            device: torch.device) -> tuple[str, float]:
    img = np.array(Image.open(image_path).convert("RGB"))
    infer_tfm = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    tensor = infer_tfm(image=img)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return classes[idx.item()], conf.item()

def main():
    parser = argparse.ArgumentParser(
        description="Inference script: predict image class using ResNet-34"
    )
    parser.add_argument("image_path", type=str, help="Путь до входного изображения")
    parser.add_argument(
        "--model_path",
        type=str,
        default="familyguyclassifier.pth",
        help="Файл с сохранённым checkpoint (model_state + classes)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="(Необязательно) Папка датасета, если нужно извлечь классы из ImageFolder"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"ERROR: файл не найден: {args.image_path}")
        exit(1)
    if not os.path.isfile(args.model_path):
        print(f"ERROR: checkpoint не найден: {args.model_path}")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, classes = load_model(args.model_path, device)

    if classes is None:
        if args.data_root is None:
            raise ValueError("Для извлечения классов из папок нужно указать --data_root")
        dataset = ImageFolder(args.data_root)
        classes = dataset.classes

    label, confidence = predict(model, classes, args.image_path, device)
    print(f"Predicted: {label} (confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()
