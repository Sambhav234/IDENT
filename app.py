import os
import tempfile
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from detector import load_model, predict_image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please choose an image file first."
        else:
            filename = secure_filename(file.filename)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    dir=app.config["UPLOAD_FOLDER"],
                    suffix=os.path.splitext(filename)[1] or ".jpg",
                ) as tmp:
                    file.save(tmp.name)
                    temp_path = tmp.name

                score = predict_image(temp_path, model=get_model())
                label = "Fake" if score >= 0.5 else "Real"
                result = {
                    "label": label,
                    "score": round(score, 3),
                    "confidence": round(abs(score - 0.5) * 2, 3),
                    "filename": filename,
                }
            except Exception as exc:
                error = f"Prediction failed: {exc}"
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
