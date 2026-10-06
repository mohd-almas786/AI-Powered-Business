import numpy as np
from PIL import Image


IMAGE_SIZE = (64, 64)


def load_and_preprocess_image(image_path):
    """
    Loads an image and prepares it for a deep learning model.

    Steps:
    1. Open image
    2. Convert to grayscale
    3. Resize to 64x64
    4. Normalize pixel values
    5. Convert to NumPy array
    6. Add channel and batch dimensions
    """

    image = Image.open(image_path)

    print("Original image size:", image.size)
    print("Original image mode:", image.mode)

    # Convert to grayscale
    image = image.convert("L")

    # Resize
    image = image.resize(IMAGE_SIZE)

    # Convert to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Normalize pixel values from 0-255 to 0-1
    image_array = image_array / 255.0

    # Shape:
    # (64, 64)
    #       ↓
    # (1, 64, 64)   -> channel
    #       ↓
    # (1, 1, 64, 64) -> batch
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=0)

    print("Processed image shape:", image_array.shape)
    print(
        "Pixel range:",
        image_array.min(),
        "to",
        image_array.max()
    )

    return image_array


if __name__ == "__main__":
    print("Computer Vision preprocessing module loaded successfully.")
    print("Expected image size:", IMAGE_SIZE)