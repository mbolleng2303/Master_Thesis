import nibabel as nib
import os
import numpy as np
from PIL import Image as im
from pathlib import Path

data_folder = Path("C:/Users/localadmin/Desktop/Maxime_Bollengier/data_test/")
image = data_folder / "lung_020_0000.nii.gz"
image = nib.load(image)

def normalize1(arr):
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
       minval = arr[..., i].min()
    maxval = arr[..., i].max()
    if minval != maxval:
       arr[..., i] -= minval
    arr[..., i] *= (255.0 / (maxval - minval))
    return arr

# z-score normalization
def normalize(arr):
    arr = np.array(arr)
    m = np.mean(arr)
    s = np.std(arr)
    return (arr - m) / s
#transform to canonical coordinates (RAS)
image = nib.as_closest_canonical(image)
#image.set_data_dtype(np.uint8)
data = image.get_fdata()

test = data[:,:,100]
test = test *-1
test2 = normalize1(normalize(test))

test1 = normalize(test)

data2 = im.fromarray(test2)
data1 = im.fromarray(test1)
data2.convert('L')
data1.convert('L')

# saving the final output
# as a PNG file
data2.save('test2.tif')
data1.save('test1.tif')
import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

img = img_as_float(gray2rgb(test2))
#img = img_as_float(astronaut()[::2, ::2])
segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=250, compactness=10, sigma=1,
                     start_label=1)
segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=250, compactness=0.001)

print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()

"""
# to be extra sure of not overwriting data:
new_data = np.copy(image.get_fdata())
hd = image.header



# update data type:
new_dtype = np.float32 # for example to cast to int8.
new_data = new_data.astype(new_dtype)
image.set_data_dtype(new_dtype)

# if nifty1
if hd['sizeof_hdr'] == 348:

    new_image = nib.Nifti1Image(new_data, image.affine, header=hd)
# if nifty2
elif hd['sizeof_hdr'] == 540:
    new_image = nib.Nifti2Image(new_data, image.affine, header=hd)
else:
    raise IOError('Input image header problem')

img_np = new_image.get_fdata()



'''print(img_np)'''
test = img_np[:,:,100]
'''print(test.shape)'''
print(test)
data = im.fromarray(test)
data.convert('L')

# saving the final output
# as a PNG file
data.save('gfg_dummy_pic.tif')"""

