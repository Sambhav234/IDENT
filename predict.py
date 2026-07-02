import os
import sys
from detector import predict_image, load_model

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py some_image.jpg")
        sys.exit(1)

    path = sys.argv[1]
    model = load_model()
    score = predict_image(path, model=model)
    print(round(score, 3))
 