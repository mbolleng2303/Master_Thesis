
import pickle
import time
import json
from scipy import sparse as sp
from batchgenerators.utilities.file_and_folder_operations import join
import dgl

from Image_Processing.preprocessing import normalize, normalize1
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
import os


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

data_dir = 'C:/Users/localadmin/OneDrive/Maxime_Bollengier - Copy/nnUNet_raw_data_base/nnUNet_raw_data/Task006_Lung'
data = create_lists_from_splitted_dataset(data_dir)
idx = []
idy = []
idz = []
Idx = []
Idy = []
Idz = []
for img, img_lab in data:
    image = nib.load(img_lab)
    # transform to canonical coordinates (RAS)
    image = nib.as_closest_canonical(image)
    image.set_data_dtype(np.uint8)
    image = np.array(image.get_fdata())
    print(image.shape)
    nonzeroind = np.nonzero(image)
    idx.append(nonzeroind[0].min())
    idy.append(nonzeroind[1].min())
    idz.append(nonzeroind[2].min())
    Idx.append(nonzeroind[0].max())
    Idy.append(nonzeroind[1].max())
    Idz.append(nonzeroind[2].max())
    print(np.array(idx).min(),np.array(idy).min(),np.array(idz).min(),np.array(Idx).max(),np.array(Idy).max(),np.array(Idz).max())


