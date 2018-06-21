
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from numpy import array, ones, zeros, arctan2, pi, hstack, sum, finfo
from sklearn.svm import LinearSVC
from scipy.ndimage import convolve
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.model_selection import cross_val_score
from PIL import Image
from skimage import filters
from sys import argv
from sklearn.cluster import DBSCAN

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# Creating creating temporary directory for frames

workdir = './frames_' + os.path.splitext(os.path.basename(argv[1]))[0]
if not os.path.exists(workdir):
    os.mkdir(workdir)

# Rescaling video
os.system("ffmpeg -i " + argv[1] + " -vf scale=320:240 ./sm_" + os.path.splitext(os.path.basename(argv[1]))[0] +
          os.path.splitext(os.path.basename(argv[1]))[1])

# Extracting frames
os.system("ffmpeg -i ./sm_" + os.path.splitext(os.path.basename(argv[1]))[0] +
          os.path.splitext(os.path.basename(argv[1]))[1] + ' ' + workdir + '/frame%d.png')

# Clustering
def backgr(frms):
    shape = (len(frms), len(frms[0]), len(frms[0][0]))
    image = []
    for i in range(shape[1]):
        row = []
        for j in range(shape[2]):
            pixels = np.array([frame[i][j] for frame in frms])
            db = DBSCAN(eps=4, min_samples=4).fit(pixels)
            labels = db.labels_
            label_set = set(labels)
            lbcnt = {k: 0 for k in label_set}
            for k in labels:
                lbcnt[k] += 1
            choice = -1
            for k in label_set:
                if choice == -1 or lbcnt[choice] < lbcnt[k] and k != -1:
                    choice = k            
            if choice != -1:
                row.append(np.mean(np.array([pixels[k] for k in range(len(labels)) if labels[k] == choice]), axis=0))
                frms[:, i, j, 0] -= frms[:, i, j, 0] * (labels == -1)
                frms[:, i, j, 2] -= frms[:, i, j, 2] * (labels == -1)
            else:
                row.append(np.mean(pixels, axis=0))
        image.append(row)
        if i % 100 == 99:
            print(i + 1)

    return image

frames = np.array([np.asarray(Image.open(os.path.join(workdir, 'frame' + str(f + 1) + '.png')))
                   for f in range(len(os.listdir(workdir)))])

image1 = frames[0]
k = 0
prev = 0
res = []
term = []

# Scene change detection
for i in range(1, len(os.listdir(workdir))):
    image2 = frames[i]
    img1 = rgb2gray(image1)
    img2 = rgb2gray(image2)
    if sum((img1 - img2)**2) > 5000:
        k += 1
        if i - prev > 10:
            if k <= 7:
            ans = backgr(frames[prev:i - 1])
            ans = np.array(ans).astype(np.uint8)
            res.append(ans)
            term.append(prev)
        prev = i - 1
    image1 = image2
print(k)
term.append(prev)
term.append(len(os.listdir(workdir)))


# ans = backgr(frames[prev:min(prev + 100, len(os.listdir(workdir)))])
ans = backgr(frames[prev:len(os.listdir(workdir))])
ans = np.array(ans).astype(np.uint8)
res.append(ans)

res = np.array(res).astype(np.uint8)
cpy = np.copy(res[0])

term = np.array(term)

frames = np.array(frames).astype(np.uint8)

# Searching for the same locations
loc = np.arange(res.shape[0])
for i in range(loc.shape[0]):
    if i == 0 or loc[i] == i:
        if i != 0 and loc[i] == i:
            loc[i] = loc[i-1] + 1
        img1 = rgb2gray(res[i])
        for j in range(i + 1, loc.shape[0]):
            img2 = rgb2gray(res[j])
            if sum((img1 - img2) ** 2) < 5000:
                loc[j] = loc[i]

outdir = './loc_' + os.path.splitext(os.path.basename(argv[1]))[0]
if not os.path.exists(outdir):
    os.mkdir(outdir)

# Adding location number to frames and saving
for i in range(1, term.shape[0]):
    for j in range(term[i - 1], term[i]):
        imsave(outdir + '/loc' + str(j + 1) + '.png', frames[j])
        img = Image.open(outdir + '/loc' + str(j + 1) + '.png')
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), "location " + str(loc[i-1] + 1), (0, 255, 255))
        img.save(outdir + '/loc' + str(j + 1) + '.png')
        
for i in range(res.shape[0]):
    imsave(outdir + '/locrs' + str(i + 1) + '.png', res[i])

# Creating final video
if len(argv) < 3:
    outpath = './movie.mp4'
else:
    outpath = argv[2]
os.system("ffmpeg -r 30 -i " + outdir + "/loc%d.png -vcodec mpeg4 -y " + outpath)

# Deleting files
os.remove("./sm_" + os.path.splitext(os.path.basename(argv[1]))[0] + os.path.splitext(os.path.basename(argv[1]))[1])
for f in os.listdir(workdir):
    os.remove(os.path.join(workdir, f))
os.rmdir(workdir)
for f in os.listdir(outdir):
    os.remove(os.path.join(outdir, f))
os.rmdir(outdir)

