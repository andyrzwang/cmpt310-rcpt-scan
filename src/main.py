import os
import sys
import cv2
import numpy as np

# Import from preprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'preprocess')))
from file_full_path import folder_file_path
from grayscale import grayscale
from binarization import apply_binarization
from fix_tilted_image import fix_tilted_image  # now from its own module

def save_step_image(img, step_name, input_filename):
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"proof_{base_name}_{step_name}.png"
    out_path = folder_file_path("dataOut", output_filename)

    if isinstance(img, np.ndarray):
        cv2.imwrite(out_path, img)
        print(f"✅ Saved: {out_path}")
    else:
        print(f"⚠️  Skipped {step_name}: invalid image (None or not a NumPy array)")

def run_pipeline(input_filename):
    image_path = folder_file_path("images", input_filename)
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    print(f"\n🔄 Processing {image_path}")
    image = cv2.imread(image_path)

    # Step 1: Grayscale
    gray = grayscale(image)
    save_step_image(gray, "grayscale", input_filename)

    # Step 2: Binarization
    binary = apply_binarization(gray)
    save_step_image(binary, "binarized", input_filename)

    # Step 3: Deskew (now returns angle and image)
    best_angle, deskewed = fix_tilted_image(binary)
    if isinstance(deskewed, np.ndarray):
        save_step_image(deskewed, "deskewed", input_filename)
        print(f"📐 Deskew angle: {best_angle:.2f} degrees")
    else:
        print("⚠️  Deskew step failed — skipping save.")

    print("✅ Preprocessing complete.")

if __name__ == "__main__":
    # Loop from 000.jpg to 020.jpg
    for i in range(21):
        filename = f"{i:03d}.jpg"
        run_pipeline(filename)
