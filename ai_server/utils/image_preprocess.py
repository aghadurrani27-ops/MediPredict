import numpy as np
from PIL import Image

def preprocess_image(image):
    image = image.resize((224,224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
