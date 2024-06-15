import os
from PIL import Image
import numpy as np

import time

startTime = time.time()

# VARIABLES
THRESHOLD = 230
WHITES_IN_THE_ROW = 100

list = os.listdir('input')

image = Image.open('input/' + list[3])
# image.show()

image_array = np.array(image)

"""
image_array[10:1000, 10:100, :] = [255, 0, 0]
im = Image.fromarray(image_array)
im.show()
"""

# Image.fromarray(image_array).show()

"""
for j in range(len(image_array[0])):
    if np.all(image_array[0][j] > 240):
        print(j)
"""

def getCols(array, threshold):
    whiteCols = []

    counter = 0

    for i in range(len(array[0])):
        counter = 0
        for j in range(len(array)):
            if np.all(array[j][i] > threshold):
                counter += 1
        if counter > WHITES_IN_THE_ROW:
            if prev_was_white or (i > 0 and i < len(array[0]) - 1 and np.all(array[j][i-1] > threshold) and np.all(array[j][i+1] > threshold)):
                whiteCols.append(i)
                prev_was_white = True
            else:
                prev_was_white = False
        else:
            prev_was_white = False

    return whiteCols

imgWithout = getCols(image_array, THRESHOLD)

print(imgWithout)

def dropCols(array, cols):
    return np.delete(array, cols, axis=1)

newImg = dropCols(image_array, imgWithout)
Image.fromarray(newImg).show()

endTime = time.time()

print("Execution time: ", endTime - startTime)


