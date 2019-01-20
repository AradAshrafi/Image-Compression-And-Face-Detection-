from PIL import Image
import os
import numpy

from lib.lib import DATASET_LOCATION


# load all images from path
def load_images_from_folder():
    images = []
    for filename in os.listdir(DATASET_LOCATION):
        path_to_file = os.path.join(DATASET_LOCATION, filename)
        # Convert Each Image to Gray Scale
        img = numpy.asarray(Image.open(path_to_file).convert("L"), dtype=numpy.float16)
        if img is not None:
            images.append(img)
    return numpy.asarray(images)
