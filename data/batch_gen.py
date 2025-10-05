import os
import cv2
import numpy as np

input_folder = "images"
output_folder = "silhouettes"

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".png")):
            # Input path
            input_path = os.path.join(root, file)
            
            # Recreate subfolder structure inside output folder
            relative_path = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)
            
            # Output path
            output_path = os.path.join(output_subfolder, file)
            
            # Read and process to grayscale
            img = cv2.imread(input_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Threshold to separate foreground and background
            _, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find largest contour by area
            if contours:  # make sure at least one was found
                largest = max(contours, key=cv2.contourArea)

                result = np.full_like(gray, 255)  # white background
                cv2.drawContours(result, [largest], -1, 0, thickness=cv2.FILLED)
            else:
                result = np.full_like(gray, 255)  # just a blank white image

            # Save
            cv2.imwrite(output_path, result)