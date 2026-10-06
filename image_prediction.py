import os
import sys
import numpy as np
import torch
from PIL import Image

# Allow Python to find the project files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NLP_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "nlp")

if NLP_DIR not in sys.path:
    sys.path.insert(0, NLP_DIR)

from cnn_model import CustomerImageCNN


IMAGE_SIZE = (64, 64)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models",
    "cnn_model.pth"
)


def preprocess_image(image_path):
    """
    Prepare an image for CNN prediction.
    """

    image = Image.open(image_path)

    image = image.convert("L")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_tensor = torch.tensor(
        image_array,
        dtype=torch.float32
    )

    return image_tensor


def load_model():
    """
    Load the trained CNN model.
    """

    model = CustomerImageCNN(
        num_classes=2
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    return model


def predict_image(image_path):
    """
    Predict the class of an image.
    """

    if not os.path.exists(image_path):
        print("Image not found:")
        print(image_path)
        return

    if not os.path.exists(MODEL_PATH):
        print("CNN model not found:")
        print(MODEL_PATH)
        return

    model = load_model()

    image_tensor = preprocess_image(
        image_path
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    if predicted_class == 0:
        prediction = "LOW-RISK PATTERN"
    else:
        prediction = "HIGH-RISK PATTERN"

    print("\n--------------------------------")
    print("Computer Vision Prediction")
    print("--------------------------------")

    print("Image:", image_path)

    print(
        "Prediction:",
        prediction
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )


if __name__ == "__main__":

    print("Computer Vision prediction module loaded successfully.")

    print("\nTrained CNN model:")
    print(MODEL_PATH)

    image_path = input(
        "\nEnter the full path of an image: "
    ).strip().strip('"')

    predict_image(image_path)