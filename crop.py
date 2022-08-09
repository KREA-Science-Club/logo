import os

import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt

def crop_to_square(filename):        
    img = image.imread(filename)

    h, w, c = img.shape

    img = img[:, w//2 - h//2:w//2 + h//2, :]
    # img = np.array([row[w//2 - h//2:w//2 + h//2] for row in img])
    
    name, ext = os.path.splitext(filename)
    cropped_name = f"{name}_square{ext}"

    plt.imsave(cropped_name, img)