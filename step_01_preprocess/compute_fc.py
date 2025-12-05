import nibabel as nib
import numpy as np
import os
import sys


def get_timeseries(func, labels):

    func_array = np.vstack([func[i].data for i in range(len(func))])
    timeseries = np.zeros((func_array.shape[0], 180)) # 180 parcels in HCP-MMP1

    for idx in range(180):
        mask = (labels == idx+1) # the 0th label is ???
        mean_ts = np.mean(func_array[:, mask], axis=1)
        timeseries[:, idx] = mean_ts

    return timeseries


def get_fc(subj):

    print(f"Subject: {subj}")

    l_parcel_path = "/home/qwer/generateFC/parcellation/lh.HCP-MMP1.annot"
    r_parcel_path = "/home/qwer/generateFC/parcellation/rh.HCP-MMP1.annot"

    l_func_path = f"/home/qwer/generateFC/fsaverage_func_gifti/{subj}.REST2_LR.L.rfMRI.fsavg_164k.func.gii"
    r_func_path = f"/home/qwer/generateFC/fsaverage_func_gifti/{subj}.REST2_LR.R.rfMRI.fsavg_164k.func.gii"

    for path in [l_func_path, r_func_path, l_parcel_path, r_parcel_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    l_func = nib.load(l_func_path)
    r_func = nib.load(r_func_path)

    l_labels, _, _ = nib.freesurfer.io.read_annot(l_parcel_path) # returns (labels, ctab, names)
    r_labels, _, _ = nib.freesurfer.io.read_annot(r_parcel_path)

    l_timeseries = get_timeseries(l_func.darrays, l_labels)
    r_timeseries = get_timeseries(r_func.darrays, r_labels)
    full_timeseries = np.hstack((l_timeseries, r_timeseries))
    fc_matrix = np.corrcoef(full_timeseries.T)

    output_path = f"{output_dir}/functional_connectivity/{subj}_fc_matrix_r2_LR.npy"
    np.save(output_path, fc_matrix)


subj = sys.argv[1]
output_dir = sys.argv[2]

get_fc(subj)