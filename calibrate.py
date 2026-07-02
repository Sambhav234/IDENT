import sys
from detector import train_model, print_evaluation

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calibrate.py <real_folder> <fake_folder>")
        sys.exit(1)

    real_folder = sys.argv[1]
    fake_folder = sys.argv[2]
    model = train_model(real_folder, fake_folder)
    print("Training complete")
    print_evaluation(real_folder, fake_folder, model)