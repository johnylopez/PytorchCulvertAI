import numpy as np
import cv2

#Creation of the output image 
def createOutputImage(pred_image, resized_image):
    rgb_imageP = np.zeros((pred_image.shape[0], pred_image.shape[1], 3), dtype=np.uint8)
    deficiencies = set()
    color_map = {
        0: [0, 0, 0],        # Background color
        1: [192, 209, 249],  # crack
        2: [191, 191, 191],  # hole
        3: [254, 255, 126],  # root
        4: [247, 205, 160],  # Deformation
        5:[246,202,255],     # Fracture
        6: [139, 2, 223],    # Erosion
        7: [188, 252, 176],  # joints
        8: [243,172,0]       #loose_gasket

    }

    tensor = pred_image.cpu()
    unique_values = np.unique(tensor)
    for value in unique_values:
        if value in color_map:
            rgb_imageP[tensor == value] = color_map[value]
            deficiencies.add({
                0: "",
                1: "crack",
                2: "hole",
                3: "root",
                4: "deformation",
                5: "fracture",
                6: "erosion",
                7: "joints",
                8: "loose gasket"
            }[value])

    output = cv2.addWeighted(resized_image, 1, rgb_imageP, 1,0)
        
    return output
