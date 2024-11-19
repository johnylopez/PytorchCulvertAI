import numpy as np
import cv2
from scipy.ndimage import binary_dilation

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

    tensor = pred_image.cpu().numpy()
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
    
    # # Create a mask for the predicted areas (non-background regions)
    # mask = (rgb_imageP != 0)

    # # Apply Canny edge detection to the mask to find the borders
    # gray_image = cv2.cvtColor(rgb_imageP, cv2.COLOR_RGB2GRAY)  # Convert RGB to grayscale for Canny
    # edges = cv2.Canny(gray_image, 100, 200)  # You can tweak the thresholds (100, 200)
    # edge_mask = edges > 0  # Convert edge result to a boolean mask

    # # Prepare the final output image
    # output = np.zeros_like(rgb_imageP)
    
    # # Apply alpha blending for the main mask (non-background regions)
    # alpha = 0.5
    # output[mask] = np.clip((alpha * rgb_imageP[mask] + (1 - alpha) * resized_image[mask]), 0, 255).astype(np.uint8)
    
    # # Set border color to red [255, 0, 0], using the edge mask
    # border_color = np.array([255, 0, 0], dtype=np.uint8)
    # output[edge_mask] = border_color  # Set the border pixels to red using the Canny edge mask
    
    # # For regions not covered by the mask or the edges, use the resized image
    # print(output[~mask & ~edge_mask].shape)
    # output[~mask & ~edge_mask] = resized_image[~mask & ~edge_mask]  






    # mask = (rgb_imageP != 0)
    # border_width = 3
    # dilated_mask = binary_dilation(mask, iterations=border_width)
    # border_mask = dilated_mask & ~mask

    # output = np.zeros_like(rgb_imageP)
    # # output[mask] = rgb_imageP[mask]
    # alpha = 0.5
    # output[mask] = (alpha * rgb_imageP[mask] + (1-alpha) * resized_image[mask]).astype(rgb_imageP.dtype)
    # border_color = np.array([255,0,0])
    # print(output[border_mask].shape)
    # # output[border_mask] = border_color.reshape(1,1,3)   
    # output[border_mask] = border_color
    

    # output[~mask & ~border_mask] = resized_image[~mask & ~border_mask]

    edges = np.zeros_like(tensor, dtype=np.uint8)
    for value in unique_values:
        if value in color_map and value != 0:  # We skip the background
            mask = (tensor == value).astype(np.uint8)
            edges = np.maximum(edges, mask)  # Create a mask of the object
    
    # Find borders of the regions
    border_edges = cv2.Canny(edges.astype(np.uint8), 100, 200)  # Find borders
    
    # Convert the border edges to RGB and apply the color for borders
    border_rgb = np.zeros_like(rgb_imageP)
    for value in unique_values:
        if value in color_map and value != 0:  # Skip the background
            border_rgb[border_edges == 255] = [255, 0, 0]  # Highlight borders in red (or any color)

    # Combine the original image with the mask and highlight borders
    blended_image = cv2.addWeighted(resized_image, 0.7, rgb_imageP, 0.5, 0)  # Transparency on the mask

    # Overlay the borders on the blended image with a stronger weight
    output = cv2.addWeighted(blended_image, 1, border_rgb, 1, 0)

    # output = cv2.addWeighted(resized_image, 1, rgb_imageP, 1,0)
        
    return output,deficiencies
