import sys
import os

# Add the path to the parent folder of preprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grayscale import grayscale
import cv2
import numpy as np

def test_grayscale():
    path = 'cmpt310-rcpt-scan/data/dirty/large-receipt-image-dataset-SRD/1000-receipt.jpg'
    
    result = grayscale(path)

    assert result is not None, "Failed to load image"
    assert len(result.shape) == 2, "Image is not in grayscale"

    print("✅ Test passed!")

if __name__ == "__main__":
    test_grayscale()
