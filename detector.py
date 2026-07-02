import os
import glob
import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 8


class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


class TorchImageClassifier:
    def __init__(self, model, transform, device):
        self.model = model
        self.transform = transform
        self.device = device

    def _predict_one(self, path):
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return np.array([probs[0], probs[1]], dtype=np.float32)

    def predict_proba(self, X):
        if isinstance(X, (str, os.PathLike)):
            return self._predict_one(str(X)).reshape(1, -1)
        if isinstance(X, (list, tuple)):
            return np.vstack([self._predict_one(str(p)) for p in X])
        raise TypeError("Expected a path or a list of paths")

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


def _is_image(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def _list_image_files(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted(glob.glob(os.path.join(folder, "*")))
    return [p for p in files if _is_image(p)]


def _get_transforms(train=False):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _build_model():
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model


def _train_model(model, loader, epochs=EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        print(f"epoch {epoch + 1}/{epochs} loss={total_loss / max(1, len(loader.dataset)):.4f}")

    return model.eval().to(device)


def train_model(real_folder, fake_folder, output_path=MODEL_PATH):
    real_files = _list_image_files(real_folder)
    fake_files = _list_image_files(fake_folder)
    if not real_files or not fake_files:
        raise ValueError("Both real and fake folders must contain images")

    all_paths = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)
    train_tfm = _get_transforms(train=True)
    dataset = ImageDataset(all_paths, labels, train_tfm)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = _build_model()
    model = _train_model(model, loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = TorchImageClassifier(model=model, transform=_get_transforms(train=False), device=device)
    joblib.dump(classifier, output_path)
    return classifier


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def predict_image(path, model=None):
    if model is None:
        model = load_model()
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([path])[0]
        return float(probs[1])
    raise TypeError("Loaded model does not support predict_proba")


def evaluate_folder(folder, model, label):
    files = _list_image_files(folder)
    if not files:
        raise ValueError(f"No image files found in {folder}")
    probs = [predict_image(path, model=model) for path in files]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    acc = accuracy_score([label] * len(preds), preds)
    return acc, probs, preds


def print_evaluation(real_folder, fake_folder, model):
    real_acc, real_probs, real_preds = evaluate_folder(real_folder, model, 0)
    fake_acc, fake_probs, fake_preds = evaluate_folder(fake_folder, model, 1)
    print("Real folder accuracy:", round(real_acc, 3))
    print("Fake folder accuracy:", round(fake_acc, 3))
    print("Real probs sample:", [round(x, 3) for x in real_probs[:5]])
    print("Fake probs sample:", [round(x, 3) for x in fake_probs[:5]])
