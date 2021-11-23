import json
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, subfiles, subdirs, isfile
from pathlib import Path
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



print(len(lst))

