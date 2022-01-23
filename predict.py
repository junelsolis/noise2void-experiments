from n2v.models import N2V
# import numpy as np
from skimage import io, img_as_float32, img_as_uint
from tifffile import imread
from csbdeep.io import save_tiff_imagej_compatible



model_name = 'n2v_3D'
basedir = 'models'
model = N2V(config=None, name=model_name, basedir=basedir)

img = imread('data/sample.tif')

tp2 = img[1]
pred = model.predict(tp2, axes='ZYX',n_tiles=(2,4,4))
# io.imsave('pred.tif', img_as_uint(pred))
save_tiff_imagej_compatible('prediction.tif', pred, 'ZYX')


# for t in range(len(img[0])):
#     pred = model.predict(img[t], axes='ZYX')
#     # plt.figure(figsize=(30,30))

#     # # We show the noisy input...
#     # plt.subplot(1,2,1)
#     # plt.imshow(np.max(img,axis=0),
#     #         cmap='magma',
#     #         vmin=np.percentile(img,0.1),
#     #         vmax=np.percentile(img,99.9)
#     #         )
#     # plt.title('Input');

#     # # and the result.
#     # plt.subplot(1,2,2)
#     # plt.imshow(np.max(pred,axis=0), 
#     #         cmap='magma',
#     #         vmin=np.percentile(pred,0.1),
#     #         vmax=np.percentile(pred,99.9)
#     #         )
#     # plt.title('Prediction');
    
#     save_tiff_imagej_compatible('prediction.tif', pred, 'ZYX')
#     break