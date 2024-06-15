import os
from PIL import Image
import numpy as np

import time


startTime = time.time()

# VARIABLES
THRESHOLD = 200
WHITES_IN_THE_ROW = 1000

list = os.listdir('input')

image = Image.open('input/' + list[3])
# image.show()

image_array = np.array(image)
# print(image_array[400])

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

THRESHOLD = 200
WHITES_IN_THE_ROW = 1500

def getRows(array):
    whiteCols = []
    counter = 0
    prev_was_white = False

    for i in range(len(array)):
        counter = 0
        for j in range(len(array[0])):
            if np.all(array[i][j] > THRESHOLD):
                counter += 1
        if counter > WHITES_IN_THE_ROW:
            if prev_was_white or (i > 0 and i < len(array[0]) - 1 and np.all(array[j][i-1] > THRESHOLD) and np.all(array[j][i+1] > THRESHOLD)):
                whiteCols.append(i)
                prev_was_white = True
            else:
                prev_was_white = False
        else:
            prev_was_white = False

    return whiteCols

cols = getRows(image_array)

def findJump(array):
    for i in range(len(array)):
        if (i + 1) != len(array) & (array[i] + 1) == array[i + 1] & i > 200:
            return array[i + 1]
        
print(findJump(cols))
        
cropped1 = image.crop((0, 0, len(image_array[0]), findJump(cols)))
cropped2 = image.crop((0, findJump(cols), len(image_array[0]), len(image_array)))

cropped1.show()
cropped2.show()