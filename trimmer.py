import os
from PIL import Image
import numpy as np

# VARIABLES
THRESHOLD = 250
WHITES_IN_THE_ROW = 1000

inputDir = 'input'
outputDir = 'output'

def getCols(array):
    whiteCols = []
    counter = 0
    prev_was_white = False

    for i in range(len(array[0])):
        # print(counter)
        counter = 0
        for j in range(len(array)):
            if np.all(array[j][i] > THRESHOLD):
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

def dropCols(array, cols):
    return np.delete(array, cols, axis=1)

def dropRows(array, cols):
    return np.delete(array, cols, axis=0)

def processFolder():
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    counter = 0
    for f in os.listdir(inputDir):
        if f.lower().endswith('.jpeg'):
            counter += 1
            path = os.path.join(inputDir, f)
            imgArr = np.array(Image.open(path))
            cols = getCols(imgArr)
            imgArr = dropCols(imgArr, cols)
            imgArr = dropRows(imgArr, getRows(imgArr))
            Image.fromarray(imgArr).rotate(90, expand=True).save(os.path.join(outputDir, f))
            print(f"Done: {counter} / {len(os.listdir(inputDir))}")


if __name__ == "__main__":
    processFolder()