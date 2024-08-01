import cv2
import numpy as np

def get_apparent_skin_color(image):
    # Assuming this function processes the image and returns a skin tone category: 'Fair', 'Medium', 'Dark', etc.
    # This is a placeholder implementation.
    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_r, avg_g, avg_b = avg_color

    if avg_r > 150 and avg_g > 130 and avg_b > 120:
        return 'Fair'
    elif avg_r > 100 and avg_g > 85 and avg_b > 75:
        return 'Medium'
    else:
        return 'Dark'
