from n2v.models import N2V
import numpy as np
from skimage import io, img_as_uint
# from tifffile import imread
# from tqdm import tqdm
# from csbdeep.io import save_tiff_imagej_compatible


model_name = 'n2v_3D'
basedir = 'models'
model = N2V(config=None, name=model_name, basedir=basedir)

img = img_as_uint(io.imread('data/sample.tif'))
img_denoised = np.copy(img)

for t in range(img.shape[0]):
    img_denoised[t] = model.predict(img[t], axes='ZYX', n_tiles=(2,4,4))
    
io.imsave('data/denoised.tif', img_as_uint(img_denoised), plugin='tifffile', check_contrast=False, **{"metadata": {"axes": "TZYX"}, "imagej": True})