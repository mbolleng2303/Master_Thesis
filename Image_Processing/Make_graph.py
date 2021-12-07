import json
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, subfiles, subdirs, isfile
from pathlib import Path
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
import pandas as pd
path = 'C:/Users/localadmin/OneDrive/Maxime_Bollengier - Copy/nnUNet_raw_data_base/nnUNet_raw_data/Task006_Lung'
def create_lists_from_splitted_dataset_folder(folder):
    """
    does not rely on dataset.json
    :param folder:
    :return:
    """
    caseIDs = get_caseIDs_from_splitted_dataset_folder(folder)
    list_of_lists = []
    for f in caseIDs:
        list_of_lists.append(subfiles(folder, prefix=f, suffix=".nii.gz", join=True, sort=True))
    return list_of_lists
def get_caseIDs_from_splitted_dataset_folder(folder):
    files = subfiles(folder, suffix=".nii.gz", join=False)
    # all files must be .nii.gz and have 4 digit modality index
    files = [i[:-12] for i in files]
    # only unique patient ids
    files = np.unique(files)
    return files
def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        cur_pat.append(join(base_folder_splitted, "labelsTr", tr['label'].split("/")[-1]))
        lists.append(cur_pat)
    return lists

lst =create_lists_from_splitted_dataset(path)

image , label = lst[0]
image = nib.load(image)
label = nib.load(label)
#transform to canonical coordinates (RAS)
image = nib.as_closest_canonical(image)
label = nib.as_closest_canonical(label)
image.set_data_dtype(np.uint8)
label.set_data_dtype(np.uint8)
image = image.get_fdata()
label = label.get_fdata()
nonzeroind = np.nonzero(label)
epsilon = 5
label = label #label[nonzeroind[0].min()-epsilon:nonzeroind[0].max()+epsilon,nonzeroind[1].min()-epsilon:nonzeroind[1].max()+epsilon,nonzeroind[2].min()-epsilon:nonzeroind[2].max()+epsilon]
image = image #image[nonzeroind[0].min()-epsilon:nonzeroind[0].max()+epsilon,nonzeroind[1].min()-epsilon:nonzeroind[1].max()+epsilon,nonzeroind[2].min()-epsilon:nonzeroind[2].max()+epsilon]
#super_label = label #slic(label,n_segments=2,compactness=50, multichannel=False)
seg = len(np.unique(image))
#super_image = slic(image,n_segments=100, compactness=100,multichannel=False)
a=4

def label_score(gt_labels, sp_segs):
    # type: (np.ndarray, np.ndarray) -> float
    """
    Score how well the superpixels match to the ground truth labels.
    Here we use a simple penalty of number of pixels misclassified
    :param gt_labels: the ground truth labels (from an annotation tool)
    :param sp_segs: the superpixel segmentation
    :return: the score (lower is better)
    """
    out_score = 0
    for idx in np.unique(sp_segs):
        cur_region_mask = sp_segs == idx
        labels_in_region = gt_labels[cur_region_mask]
        labeled_regions_inside = np.unique(labels_in_region)
        if len(labeled_regions_inside) > 1:
            out_score += np.count_nonzero(labels_in_region)
            """print('Superpixel id', idx, 'regions', len(labeled_regions_inside))
            print('\n', pd.value_counts(labels_in_region))
            print('Missclassified Pixels:', np.sum(pd.value_counts(labels_in_region)[1:].values))"""
        #out_score += np.sum(pd.value_counts(labels_in_region)[1:].values)
    #print('Missclassified Pixels:', label_score(label, super_image))
    return out_score


def find_super_pixel_param (image , label , n_segments , compactness):
    best_param = []
    score = 10000000000
    for n_segment in n_segments :
        for compact in compactness :
            super_image = slic(image, n_segments=n_segment, compactness=compact, multichannel=False)
            res = label_score(label, super_image)
            if res < score :
                score = res
                best_param.append([n_segment,compact, score])
                print(best_param[-1])
    return best_param

print (find_super_pixel_param(image , label , [1000, 2000, 3000, 3900], [100,  1000, 10000]))

"""fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
ax[0, 0].imshow(mark_boundaries(image[:,:,45],super_image[:,:,45]))#mark_boundaries(img, segments_fz)
ax[0, 0].set_title("original")
ax[0, 1].imshow(label[:,:,45])
ax[0, 1].set_title('label')
ax[1, 0].imshow(super_image[:,:,45])
ax[1, 0].set_title('orginal_slic')
ax[1, 1].imshow((mark_boundaries(label[:,:,45],super_image[:,:,45])))
ax[1, 1].set_title('label_slic')

'''for a in ax.ravel():
    a.set_axis_off()'''

#plt.tight_layout()
plt.show()"""






