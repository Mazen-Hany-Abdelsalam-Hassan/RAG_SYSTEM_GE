import torch
from PIL import Image
import numpy as np
import cv2
import pytesseract
from src.config import  ID_TO_NAMES , DEBUG_DIR,SUPPRESS_THRESHOLDS 


def colormap(N=256, normalized=False):
    """
    Generate the color map.
    Args:
        N (int): Number of labels (default is 256).
        normalized (bool): If True, return colors normalized to [0, 1]. Otherwise, return [0, 255].
    Returns:
        np.ndarray: Color map array of shape (N, 3).
    """

    def bitget(byteval, idx):
        """
        Get the bit value at the specified index.
        Args:
            byteval (int): The byte value.
            idx (int): The index of the bit.
        Returns:
            int: The bit value (0 or 1).
        """
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << (7 - j))
            g = g | (bitget(c, 1) << (7 - j))
            b = b | (bitget(c, 2) << (7 - j))
            c = c >> 3
        cmap[i] = np.array([r, g, b])

    if normalized:
        cmap = cmap.astype(np.float32) / 255.0

    return cmap


def visualize_bbox(image_path, bboxes, classes, scores, id_to_names ,alpha=0.3 ):
    """
    Visualize layout detection results on an image.
    Args:
        image_path (str): Path to the input image.
        bboxes (list): List of bounding boxes, each represented as [x_min, y_min, x_max, y_max].
        classes (list): List of class IDs corresponding to the bounding boxes.
        id_to_names (dict): Dictionary mapping class IDs to class names.
        alpha (float): Transparency factor for the filled color (default is 0.3).
    Returns:
        np.ndarray: Image with visualized layout detection results.
    """
    # Check if image_path is a PIL.Image.Image object


    if isinstance(image_path, Image.Image) or isinstance(image_path, np.ndarray):
        image = np.array(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    else:
        image = cv2.imread(image_path)

    overlay = image.copy()

    cmap = colormap(N=len(id_to_names), normalized=False)

    # Iterate over each bounding box
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = map(int, bbox)
        class_id = int(classes[i])
        class_name = id_to_names[class_id]

        text = class_name + f":{scores[i]:.3f}"

        color = tuple(int(c) for c in cmap[class_id])
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Add the class name with a background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(image, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
        cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image




def chunk_the_page(image, bboxes, classes):
    """
    Visualize layout detection results on an image.
    Args:
        image (str): pillow_image
        bboxes (list): List of bounding boxes, each represented as [x_min, y_min, x_max, y_max].
        classes (list): List of class IDs corresponding to the bounding boxes.

    Returns:
        np.ndarray: Image with visualized layout detection results.
    """
# Check if image_path is a PIL.Image.Image object
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    text_chunks = []
    table_chunk =[]
    # Iterate over each bounding box
    for i, bbox in enumerate(bboxes):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        x_min, y_min, x_max, y_max = map(int, bbox)
        class_id = int(classes[i])
        class_name =ID_TO_NAMES[class_id]
        section = image[y_min:y_max + 1, x_min:x_max + 1]
        image_result = {"text_chunk_image":section , "type": class_name}
        if class_name not in ["Table","Picture"]:
            text = pytesseract.image_to_string(section)
            text_result = {"text_chunk":text, "type":class_name}
            text_chunks.append(text_result)

        if class_name == 'Table':
            #table_chunk.append()
            pass



    return text_chunks , table_chunk

def Intersection(box1, box2):
    box1_x1 , box1_y1 , box1_x2,box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2 , box2_x2)
    y2 = torch.min(box1_y2 , box2_y2)
    intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0)
    return intersection


def Area(box):
    """
    Calculate the area of the bounding box given its coordinates
    :param box: [xmin , ymin, xmax,ymax] ,
    :return: a
    """
    return abs((box[0] - box[2]) * (box[1] - box[3]))


def Suppress_Intersect(bbox_int):
    """
     This function merge the bounding box which locate inside each other into a one box
    :param bbox_int: tensor consists of the bounding boxes ,
    each bounding box consists of [xmin , ymin , xmax , ymax]

    :return: The index of unique bounding box
    """
    keep_box = len(bbox_int) * [True]
    for i in range(len(bbox_int)):
        for j in range(i + 1, len(bbox_int)):
            area1 = Area(bbox_int[i])
            area2 = Area(bbox_int[j])
            intersection = Intersection(bbox_int[i], bbox_int[j])
            if intersection:


                if abs(area1 - intersection) / max(area1, 1) <= SUPPRESS_THRESHOLDS :
                    keep_box[i] = False
                elif abs(area2 - intersection) / max(area2, 1) <= SUPPRESS_THRESHOLDS :
                    keep_box[j] = False
    return  keep_box