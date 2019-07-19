#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import skimage
from skimage.color import rgb2hed
from matplotlib.colors import LinearSegmentedColormap
from skimage.exposure import rescale_intensity
import os
import staintools
from optparse import OptionParser
from PIL import Image
import openslide
from openslide.lowlevel import *
from openslide.lowlevel import _convert
import subprocess
import multiprocessing as mp
from joblib import Parallel, delayed
from time import sleep
import sys


# In[2]:


def main():
    parser = OptionParser()
    parser.add_option("-s", "--size", nargs = 1,type="int", default=224, help="Tile size, the default is 224")
    parser.add_option("-b", "--blank", nargs = 1,type="float", default=0.5, help="The ratio of blank area, the default is 0.5")
    parser.add_option("-w", "--window", nargs = 1, type="float", default=0.25, help="The ratio of overlapping area with the adjacent tile, the default is 0.25")
    parser.add_option("-t", "--thread", nargs = 1, type="int", default=1, help="The number of threads, the default is 1")
    parser.add_option("-o", "--output", nargs = 1, default=".", help="The path of output folder, the default is the current folder")
    parser.add_option("-i", "--input", nargs = 1, help="The path of input folder, the default is the current folder")
    parser.add_option("-n", "--normalize", nargs = 1, help="The path of the normalization target")
    parser.add_option("-l", "--level", nargs = 1,  type="int", default=0,  help="The svs level which the whole process is applied to. The default is 0 which is the one with the highest resolution")
    parser.add_option("-a", "--augment", nargs = 1, type="int", default=0, help="The number of augmented images. The defalut is 0, which means no augmented image is produced")

    (options, args) = parser.parse_args()
    tile_size = int(options.size)
    blank_ratio = float(options.blank)
    overlapping = float(options.window)
    thread = int(options.thread)
    out_dir = options.output
    folder = options.input
    normalize_target = options.normalize
    augment = options.augment
    level = options.level
    preprocess_tcga_crc_heslides(folder, out_dir, normalize_target, thread, blank_ratio, tile_size, overlapping, augment, level)


# In[ ]:


#preprocess_tcga_crc_heslides(folder, "/mnt/d/home/Yao/HE/CC_ROI_result", "/mnt/d/home/Yao/tcga_result/i1.png")


# In[ ]:


folder = "/mnt/d/home/Yao/HE/CC_ROI_svs"
target = staintools.read_image("/mnt/d/home/Yao/tcga_result/i1.png")
def preprocess_tcga_crc_heslides(folder, out_dir, normalize_target, thread = 1, blank_ratio = 0.5, tile_size = 224, overlapping = 0.25,                                 augment = 0, level = 0):
    """
    Split the entire H&E slides into small effective tiles

    Parameter: 1. folder: name of the folder where the targer image exists.
               2. out_dir: name of the output folder.
               3. normalize_target: reference image used to normalize the data
               4. thread: number of threads
               4. blank_ratio: ratio of the blank area (R > 220).
               5. tile_size: size of each tile.
               6. overlapping: step size


    Precondition: 1. folder, fileNames and out_dir are UNIX style


    """
    target = staintools.read_image(normalize_target)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)

    #for folder1 in os.listdir(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(folder + "/" + f)]
    #for file2 in files:

    #    output_dir = split_norm(folder, file2, out_dir, normalizer)



    for file2 in files:

        split_norm(folder, file2, out_dir, normalizer, overlapping, blank_ratio, tile_size, augment, level, thread)


# In[ ]:




def split_norm(folder, fileNames, out_dir, normalizer, overlapping, blank_ratio = 0.5, tile_size = 224, augment = 0, level = 0, thread=1):
    """
    Extract ROI from a entire tissue stain and normalize the ROI using vahadane and specified reference

    Args:    1. folder (str): name of the folder where the targer image exists.
             2. fileNames (str): name of the files.
             3. out_dir (str): name of the output folder.

    Precondition: 1. folder, fileNames and out_dir are UNIX style

    Return: the location of output directory

    """

#     try:
#         thresh = cv2.imread(folder + fileNames, 0)
#     except:
#         Image.MAX_IMAGE_PIXELS = 2 << 33
#         thresh = Image.open(folder + fileNames).convert("L")
#         thresh = np.array(thresh)
    print("Begin processing " + fileNames)
    if(folder[-1] == "/"):
        img = openslide.OpenSlide(folder + fileNames)
    else:
       # print(folder + "/" + fileNames)
        img = openslide.OpenSlide(folder + "/" + fileNames)
    thresh = img.read_region((0, 0), level=img.level_count-1, size = img.level_dimensions[img.level_count-1]).convert("L")
    thresh = np.array(thresh)


    ret,thresh = cv2.threshold(thresh,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    im_floodfill = thresh.copy()


    h, w = thresh.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    thresh = thresh | im_floodfill_inv
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    print("Finish threshing")
    L = skimage.measure.label(thresh, neighbors = 8)
    stats = skimage.measure.regionprops(L)
    del thresh
    del im_floodfill_inv

    print("Got the edges")


#     try:
#         A = cv2.imread(folder + fileNames)
#     except:
#         Image.MAX_IMAGE_PIXELS = 2 << 33
#         A = Image.open(folder + fileNames).convert('RGB')
#         A = np.array(A)
#         A = A[:, :, ::-1].copy()

    l_ratio = img.level_dimensions[0][0]/img.level_dimensions[img.level_count-1][0]
    w_ratio = img.level_dimensions[0][1]/img.level_dimensions[img.level_count-1][1]


    del L
#     if "tumor" in folder:
#         home_dir = (out_dir + "/tumor/" + fileNames[:fileNames.rfind(".")])
#     elif "normal" in folder:
#         home_dir = (out_dir + "/normal/" + fileNames[:fileNames.rfind(".")])
#     else:
#        home_dir = (out_dir + "/" + fileNames[:fileNames.rfind(".")])
    home_dir = (out_dir + "/" + fileNames[:fileNames.rfind(".")])
    if not os.path.isdir(home_dir):
        os.makedirs(home_dir)
    count = 0
    for i in range(len(stats)):
        s2 = stats[i].bbox
        #print(abs(s2[0]-s2[2]) > 1000 | abs(s2[1]-s2[3]) > 1000)

        if((l_ratio * abs(s2[0]-s2[2]) > 2000 )| (w_ratio * abs(s2[1]-s2[3]) > 2000)):
            #A1 = A[l_ratio * s2[0]:l_ratio * s2[2], w_ratio * s2[1]: w_ratio * s2[3],:]

            l = abs(l_ratio * s2[0] - l_ratio * s2[2]) + 1
            w = abs(w_ratio * s2[1] - w_ratio * s2[3]) + 1

            if (l * w) >= 2**29:
                openslide.lowlevel._load_image = _load_image_morethan_2_29
            else:
                openslide.lowlevel._load_image = _load_image_lessthan_2_29



            A1 = np.array(img.read_region((int(s2[1]*l_ratio), int(s2[0]*w_ratio)), level=level, size = (int(w),int(l))).convert('RGB'))
            count += 1
            print("Split " + str(count))
            split_into_tiles(home_dir, fileNames[0:9] + "_" + str(count), A1, count, normalizer, blank_ratio, tile_size, overlapping, augment, thread)
           # name = home_dir + "/" + fileNames[0:9] + "_" + str(count) + ".jpg"
           # cv2.imwrite(name, A1)
    print("Store the outputs into " + home_dir + " !")
    #return home_dir



def split_into_tiles(home_dir, fileNames, img_mat, count, normalizer, blank_ratio = 0.5, tile_size = 224, overlapping = 0.25, augment = 0, thread = 1):
    """
    Split a tissue into non-overlapping small tiles

    Args: 1. folder (str): name of the folder where the targer image exists.
          2. fileNames (str): name of the files.
          3. blank_ratio (float): ratio of the blank area (R > 220).
          4. tile_size (int): size of each tile.
          5. overlapping (float): the portion of overlapping side between two consecutive sliding windows


    Precondition: 1. folder and fileNames are UNIX style
                  2. blank_ratio is float between 0 to 1

    """
    #Need to consider the overlapping case



    img = normalizer.transform(img_mat) #normalize

    h, w, channels = img.shape
    height=tile_size + 1
    width=tile_size + 1

    h_val=height*(1 - overlapping)
    w_val=width*(1-overlapping)
    max_row = (h-height)/h_val+1
    max_col = (w-width)/w_val+1

    if max_row == np.fix(max_row):
        max_row = int(max_row)
    else:
        max_row = int(np.fix(max_row+1))

    if max_col == np.fix(max_col):
        max_col = int(max_col)
    else:
        max_col = int(np.fix(max_col+1))

    seg = np.ndarray(shape = (max_row, max_col), dtype = np.ndarray)
    loc = np.ndarray(shape = (max_row, max_col), dtype = np.ndarray)
    for row in range(1, max_row + 1):
        for col in range(1, max_col + 1):
            if ((width+(col-1)*w_val) > w) & (((row-1)*h_val+height) <= h):
                seg[row-1, col-1]= img[int((row-1)*h_val+1) : int(height+(row-1)*h_val),  int((col-1)*w_val+1) : w, : ]
                loc[row-1, col-1] = [int((row-1)*h_val+1), int(height+(row-1)*h_val), int((col-1)*w_val+1), w]

            elif ((height + (row - 1) * h_val) > w) & (((col - 1) * w_val + width) <= h):
                seg[row-1, col-1]= img[int((row-1) * h_val + 1) : int(h),  int((col-1)*w_val+1) : int(width+(col-1)*w_val), : ]
                loc[row-1, col-1] = [int((row-1) * h_val + 1), int(h), int((col-1)*w_val+1), int(width+(col-1)*w_val)]

            elif ((width + (col-1)*w_val) > w)  & (((row-1)*h_val+height) > h):
                seg[row-1, col-1] = img[int((row-1)*h_val+1) : int(h),  int((col-1)*w_val+1) : int(w), :]
                loc[row-1, col-1] = [int((row-1)*h_val+1), int(h), int((col-1)*w_val+1),  int(w)]
            else:
                seg[row-1, col-1]= img[int((row-1)*h_val+1) : int(height+(row-1)*h_val),  int((col-1)*w_val+1) : int(width+(col-1)*w_val), :]

                loc[row-1, col-1] = [int((row-1)*h_val+1), int(height+(row-1)*h_val), int((col-1)*w_val+1), int(width+(col-1)*w_val)]

    # save
    print("Begin segmenting")
    if thread > 1:
        Parallel(thread)(delayed(save_segmentation)(i, max_row, seg, tile_size, blank_ratio, home_dir, count, fileNames) for i in max_row)
    else:
        for i in max_row:
            save_segmentation(i, max_row, seg, tile_size, blank_ratio, home_dir, count, fileNames)




def save_segmentation(i, max_row, seg, tile_size = 224, blank_ratio = 0.5, home_dir, count, fileNames):
    for j in range(max_col):
            aaa = seg[i, j]
            ccc = np.shape(aaa)
            if ccc[0] == tile_size & ccc[1] == tile_size:
                if (np.sum(seg[i, j][:, :, 0] > 220)/(ccc[0] * ccc[1])) < blank_ratio:
                 #   print("start saving \n")
                    output_dir = (home_dir + "/" + str(count) + "/")
                    if not os.path.isdir(output_dir):
                        os.makedirs(output_dir)
                    cv2.imwrite( output_dir + "/" + fileNames[:fileNames.rfind(".")] + "_" + str(count) + "_" + '_'.join(map(str,loc[i, j])) + ".jpg", seg[i, j])

                    #augment
                    if augment > 0:
                        augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)
                        augmentor.fit(seg[i, j])

                        for index in range(augment):
                            augmented_image = augmentor.pop()
                            cv2.imwrite( output_dir + "/" + "aug_" + str(index) + "_" + fileNames[:fileNames.rfind(".")] + "_" + str(count) + "_"                                     '_'.join(map(str,loc[i, j])) + ".jpg", augmented_image)



# In[ ]:





def _load_image_lessthan_2_29(buf, size):
    '''buf must be a mutable buffer.'''
    _convert.argb2rgba(buf)
    return PIL.Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)


def _load_image_morethan_2_29(buf, size):
    '''buf must be a buffer.'''

    # Load entire buffer at once if possible
    MAX_PIXELS_PER_LOAD = (1 << 29) - 1
    # Otherwise, use chunks smaller than the maximum to reduce memory
    # requirements
    PIXELS_PER_LOAD = 1 << 26

    def do_load(buf, size):
        '''buf can be a string, but should be a ctypes buffer to avoid an
        extra copy in the caller.'''
        # First reorder the bytes in a pixel from native-endian aRGB to
        # big-endian RGBa to work around limitations in RGBa loader
        rawmode = (sys.byteorder == 'little') and 'BGRA' or 'ARGB'
        buf = PIL.Image.frombuffer('RGBA', size, buf, 'raw', rawmode, 0, 1)
        # Image.tobytes() is named tostring() in Pillow 1.x and PIL
        buf = (getattr(buf, 'tobytes', None) or buf.tostring)()
        # Now load the image as RGBA, undoing premultiplication
        return PIL.Image.frombuffer('RGBA', size, buf, 'raw', 'RGBa', 0, 1)

    # Fast path for small buffers
    w, h = size
    if w * h <= MAX_PIXELS_PER_LOAD:
        return do_load(buf, size)

    # Load in chunks to avoid OverflowError in PIL.Image.frombuffer()
    # https://github.com/python-pillow/Pillow/issues/1475
    if w > PIXELS_PER_LOAD:
        # We could support this, but it seems like overkill
        raise ValueError('Width %d is too large (maximum %d)' %
                         (w, PIXELS_PER_LOAD))
    rows_per_load = PIXELS_PER_LOAD // w
    img = PIL.Image.new('RGBA', (w, h))
    for y in range(0, h, rows_per_load):
        rows = min(h - y, rows_per_load)
        if sys.version[0] == '2':
            chunk = buffer(buf, 4 * y * w, 4 * rows * w)
        else:
            # PIL.Image.frombuffer() won't take a memoryview or
            # bytearray, so we can't avoid copying
            chunk = memoryview(buf)[y * w:(y + rows) * w].tobytes()
        img.paste(do_load(chunk, (w, rows)), (0, y))
    return img

if __name__ == '__main__':
    main()
