import h5py
import numpy as np
import os
from tqdm import tqdm


def get_sc(sctype='10^Fpt'):

    print(f"Type: {sctype}")

    h5_path = os.path.join(script_dir, "sc_matlab", f"individualConnectivity_{sctype}.mat")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Required file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if sctype == '10^Fpt':
            subjects = [str(subject) for subject in f['subjectIDs'][0]]
            for i, subject in tqdm(enumerate(subjects), total=len(subjects)):
                sc = f['individualConnectivity'][i].T
                np.save(os.path.join(script_dir, "structural_connectivity", sctype, f"{subject}_sc_matrix.npy"), sc)
        elif sctype == 'rawStreamlineCount':
            subjects = [str(subject) for subject in f['subjectIDs'][0]]
            for i, subject in tqdm(enumerate(subjects), total=len(subjects)):
                sc = f['rawStreamlineCounts'][i].T
                np.save(os.path.join(script_dir, "structural_connectivity", sctype, f"{subject}_sc_matrix.npy"), sc)


script_dir = os.path.dirname(os.path.abspath(__file__))

# available types: '10^Fpt', 'rawStreamlineCount', 'tractLength'

get_sc('10^Fpt')

# get_sc('rawStreamlineCount')

print("=== Process completed successfully! ===")