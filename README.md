# IDENT - Real vs Fake Image Detector

IDENT is a simple web-based image classification project that predicts whether an uploaded image looks like a real photo captured by a phone or a fake image captured from a screen such as a TV, laptop, monitor, or another digital display.

The project combines:
- a Python backend
- a Flask web interface
- a trained machine learning classifier
- image feature extraction based on computer vision signals

## Project Goal

The goal of this project is to detect whether an image is:
- Real: a photo taken directly from the real object using a phone camera
- Fake: a photo of a screen showing the object, often containing artifacts that indicate it is not a direct real-world capture

This is useful for image authenticity checks, fraud detection, and media verification tasks.

---

## How the Project Works

### 1. Image Upload
A user uploads an image through the web interface.

### 2. Feature Extraction
The uploaded image is processed using OpenCV and a set of handcrafted image features such as:
- sharpness / blur characteristics
- saturation and brightness variation
- edge density
- highlight / glare behavior
- frequency-domain periodicity signals
- color-channel differences

These features help capture visual clues that often appear in screen-captured or display-recaptured images.

### 3. Prediction
The extracted feature vector is passed into a trained classifier saved in the model file.

The classifier outputs a score between 0 and 1:
- closer to 0: the system predicts the image is Real
- closer to 1: the system predicts the image is Fake

### 4. Web Result Display
The Flask app shows:
- the prediction label
- the fake score
- confidence level

---

## Model Details

The current model is a lightweight machine learning pipeline built from:
- OpenCV-based feature extraction
- NumPy-based image statistics
- scikit-learn classifier

### Current Implementation
The project uses a trained classifier from the model file:
- [model.joblib](model.joblib)

### Why this works
Screen-captured images often contain subtle artifacts that are different from direct real-world photographs. These may include:
- periodic patterns from display pixel grids
- slight color channel misalignment
- glare and highlight artifacts
- unnatural edge and texture characteristics

The model learns which combination of these cues is most associated with fake screen-based images.

### Important Note
This is a practical first version. For very high accuracy in real-world use, the project can later be improved by:
- collecting a larger and more diverse dataset
- using a deep learning model like ResNet, MobileNet, or EfficientNet
- training with paired real-vs-screen samples under similar conditions

---

## Project Structure

```text
IDENT/
│
├── app.py                  # Flask web application
├── detector.py             # feature extraction + model training/prediction
├── train.py                # training script
├── predict.py              # command-line prediction script
├── calibrate.py            # compatibility wrapper for training
├── model.joblib            # trained model file
├── requirements.txt       # Python dependencies
├── render.yaml             # Render deployment configuration
├── .gitignore              # files ignored by Git
├── templates/
│   └── index.html          # frontend page
├── real/                   # real image examples
├── fake/                   # fake image examples
├── test_real/              # real test images
├── test_fake/              # fake test images
└── README.md               # project documentation
```

---

## Installation

### 1. Clone the project

```bash
git clone <your-repo-url>
cd IDENT
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Locally

### Start the web app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

### Run prediction from the command line

```bash
python predict.py path/to/image.jpg
```

### Train the model again

```bash
python train.py real fake
```

---

## Frontend Usage

The frontend is a simple web page where a user can:
- choose an image file
- upload it
- receive a prediction result

The page displays:
- predicted class: Real or Fake
- fake score
- confidence score

---

## Deployment on Render

This project is prepared for deployment on Render.

### Render setup

Use the following settings in Render:
- Build Command:
  ```bash
  pip install -r requirements.txt
  ```
- Start Command:
  ```bash
  gunicorn app:app
  ```

The project includes [render.yaml](render.yaml) to help with deployment configuration.

---

## Dependencies

The project uses the following Python libraries:
- Flask
- NumPy
- Pillow
- OpenCV
- scikit-learn
- joblib
- gunicorn

---

## Future Improvements

To improve accuracy and robustness, the next steps could be:
- collect more training images from different devices
- include more varied lighting conditions
- use paired real/fake samples of the same subject
- train a deep learning classifier
- improve the UI with image preview and better result visuals

---

## Summary

IDENT is a practical image authenticity detector that tries to distinguish between:
- photos captured directly from the real world
- images captured from a digital display

It uses a trained classifier and a simple Flask web app so users can upload an image and view the prediction instantly.
