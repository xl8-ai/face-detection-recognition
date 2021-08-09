from PIL import Image
import numpy as np


def expand2square(pil_img: Image, background_color: tuple):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def resize_square_image(img: Image, width: int = 640, background_color: tuple = (0, 0, 0)):
    if img.mode != 'RGB':
        return None
    img = expand2square(img, (0, 0, 0))
    img = img.resize((width, width))

    return img


def get_original_xy(xy: tuple, image_size_original: tuple, image_size_new: tuple) -> tuple:
    """Every point is from the PIL Image point of view.

    Note that image_size_original and image_size_new are PIL Image size values.
    """
    x, y = xy[0], xy[1]

    width_original = image_size_original[0]
    height_original = image_size_original[1]

    width_new = image_size_new[0]
    height_new = image_size_new[1]

    if width_original > height_original:

        x = width_original / width_new * x
        y = width_original / width_new * y

        bias = (width_original - height_original) / 2
        y = y - bias

    else:

        x = height_original / height_new * x
        y = height_original / height_new * y

        bias = (height_original - width_original) / 2
        x = x - bias

    return x, y


def get_original_bbox(bbox: np.ndarray, image_size_original: tuple, image_size_new: tuple) -> np.ndarray:
    """Get the original coordinates of the bounding box.

    Note that image_size_original and image_size_new are PIL Image size values.
    """
    bbox_new = []
    for xy in [bbox[:2], bbox[2:]]:
        xy = xy[0], xy[1]
        xy = get_original_xy(xy, image_size_original, image_size_new)
        bbox_new.append(xy)

    bbox_new = [bar for foo in bbox_new for bar in foo]
    bbox_new = np.array(bbox_new)

    return bbox_new


def get_original_lm(lm: np.ndarray, image_size_original: tuple, image_size_new: tuple) -> np.ndarray:
    """Get the original coordinates of the five landmarks.

    Note that image_size_original and image_size_new are PIL Image size values.
    """
    lm_new = []
    for lm_ in lm:
        xy = lm_[0], lm_[1]
        xy = get_original_xy(xy, image_size_original, image_size_new)
        lm_new.append(xy)

    lm_new = np.array(lm_new)

    return lm_new
