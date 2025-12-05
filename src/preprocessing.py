import cv2
import numpy as np


def load_image_from_bytes(file_bytes: bytes):
    file_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    return img


def preprocess_for_ocr(img_bgr):
    """
    Light but robust preprocessing:
    - upscale
    - grayscale
    - denoise
    - contrast boost
    - sharpen
    """
    # upscale
    img = cv2.resize(img_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # denoise but preserve edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # slight sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    return sharp
