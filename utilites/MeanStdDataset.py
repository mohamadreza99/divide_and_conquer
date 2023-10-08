import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
from config import DATA_PATH

images_path = Path(DATA_PATH + '/meta_dsets/cars/images_background')
image_train = list(images_path.glob('*/*.jpg')) # len: 101379
images_path = Path(DATA_PATH + '/meta_dsets/cars/images_evaluation')
image_eval = list(images_path.glob('*/*.jpg')) # len: 7052
image_train.extend(image_eval) # len: 108431

files = image_train

mean = np.array([0.,0.,0.])
stdTemp = np.array([0.,0.,0.])
std = np.array([0.,0.,0.])

numSamples = len(files)

for i in tqdm(range(numSamples)):
    im = cv2.imread(str(files[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.

    for j in range(3):
        mean[j] += np.mean(im[:,:,j])

mean = (mean/numSamples)

for i in tqdm(range(numSamples)):
    im = cv2.imread(str(files[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.
    for j in range(3):
        stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])

std = np.sqrt(stdTemp/numSamples)

print(mean)
print(std)


