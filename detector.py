import os
import glob
import joblib
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def _list_image_files(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted(glob.glob(os.path.join(folder, "*")))
    return [p for p in files if _is_image(p)]


def _resize_image(img, size=(224, 224)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _histogram_features(channel, bins=16):
    hist = cv2.calcHist([channel], [0], None, [bins], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    return hist.astype(np.float32)


def _fft_ring_energy(gray):
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    rmax = min(cy, cx)
    ring_mask = (r > 0.15 * rmax) & (r < 0.6 * rmax)
    if not np.any(ring_mask):
        return 0.0
    ring_vals = mag[ring_mask]
    baseline = ring_vals.mean()
    peak = ring_vals.max()
    return float((peak - baseline) / (baseline + 1e-6))


def _color_fringe_score(img):
    b, g, r = cv2.split(img.astype(np.float32))

    def high_freq(channel):
        return channel - cv2.GaussianBlur(channel, (0, 0), 2)

    hb, hg, hr = high_freq(b), high_freq(g), high_freq(r)
    return float(np.mean(np.abs(hb - hg)) + np.mean(np.abs(hg - hr)) + np.mean(np.abs(hb - hr)))


def _glare_fraction(gray):
    return float(np.mean(gray > 240))


def _edge_density(gray):
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean() / 255.0)


def _gradient_energy(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.mean())


def extract_features(img):
    img = _resize_image(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sat_std = float(hsv[:, :, 1].std())
    val_std = float(hsv[:, :, 2].std())
    bright_mean = float(gray.mean())
    contrast = float(gray.std())
    edge_density = _edge_density(gray)
    gradient_energy = _gradient_energy(gray)
    glare_fraction = _glare_fraction(gray)
    fft_ring = _fft_ring_energy(gray.astype(np.float32))
    fringe = _color_fringe_score(img)

    bgr_hist = np.concatenate([
        _histogram_features(img[:, :, 0], bins=16),
        _histogram_features(img[:, :, 1], bins=16),
        _histogram_features(img[:, :, 2], bins=16),
    ])
    hsv_hist = np.concatenate([
        _histogram_features(hsv[:, :, 0], bins=16),
        _histogram_features(hsv[:, :, 1], bins=16),
        _histogram_features(hsv[:, :, 2], bins=16),
    ])

    stats = np.array([
        lap_var,
        sat_std,
        val_std,
        bright_mean,
        contrast,
        edge_density,
        gradient_energy,
        glare_fraction,
        fft_ring,
        fringe,
    ], dtype=np.float32)

    return np.concatenate([stats, bgr_hist, hsv_hist]).astype(np.float32)


def load_feature_matrix(folder, label):
    files = _list_image_files(folder)
    if not files:
        raise ValueError(f"No image files found in {folder}")

    X, y = [], []
    for path in files:
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("could not read image")
            X.append(extract_features(img))
            y.append(label)
        except Exception as exc:
            print(f"skip {path}: {exc}")
    if not X:
        raise ValueError(f"No usable images found in {folder}")
    return np.vstack(X), np.array(y, dtype=np.int32)


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


def train_model(real_folder, fake_folder, output_path=MODEL_PATH):
    X_real, y_real = load_feature_matrix(real_folder, 0)
    X_fake, y_fake = load_feature_matrix(fake_folder, 1)

    X = np.vstack([X_real, X_fake])
    y = np.concatenate([y_real, y_fake])

    model = build_model()
    model.fit(X, y)

    joblib.dump(model, output_path)
    return model


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def predict_image(path, model=None):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if model is None:
        model = load_model()
    feats = extract_features(img)
    prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
    return prob


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
