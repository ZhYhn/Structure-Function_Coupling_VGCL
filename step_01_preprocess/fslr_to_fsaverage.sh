#!/bin/bash

# fs_LR to fsaverage conversion script
# for matching with the parcellation in fsaverage space
# usage: ./fslr_to_fsaverage.sh <subject_id> <target_resolution> <output_directory>

if [ $# -lt 3 ]; then
    echo "Usage: $0 <subject_id> <target_resolution> <output_directory>"
    echo "Example: $0 100206 164 /home/qwer/generateFC/fsaverage_func_gifti"
    exit 1
fi

SUBJECT=$1
TARGET_RES=$2
OUTPUT_DIR=$3

FMRI_DIR="/mnt/f/fMRI/${SUBJECT}_3T_rfMRI_REST_fix" # fMRI data directory
MRI_DIR="/mnt/f/MRI/${SUBJECT}_3T_Structural_preproc" # MRI data directory
STANDARD_MESHES="/home/qwer/generateFC/standard_mesh_atlases" # Standard mesh atlases directory
TMP_DIR="/home/qwer/generateFC/temp" # Temporary directory for intermediate files
RUNS=("REST2_LR") # RUNS=("REST1_LR" "REST1_RL" "REST2_LR" "REST2_RL")

mkdir -p "${TMP_DIR}"

echo "Subject: ${SUBJECT}, Target resolution: ${TARGET_RES}k"

# Step 1: Create the fsaverage-registered individual native sphere
for HEMI in L R; do
    wb_command -surface-sphere-project-unproject \
        "${MRI_DIR}/${SUBJECT}.${HEMI}.sphere.MSMAll.native.surf.gii" \
        "${STANDARD_MESHES}/fsaverage.${HEMI}_LR.spherical_std.164k_fs_LR.surf.gii" \
        "${STANDARD_MESHES}/resample_fsaverage/fs_LR-deformed_to-fsaverage.${HEMI}.sphere.164k_fs_LR.surf.gii" \
        "${TMP_DIR}/${SUBJECT}.${HEMI}.sphere.fsaverage.native.surf.gii"
    
    if [ $? -ne 0 ]; then
        echo "Error: Step 1 failed (hemisphere: ${HEMI})"
        rm -rf ${TMP_DIR}
        exit 1
    fi
done

# Step 2: Resample the individual's midthickness surface to fsaverage mesh (for area correction)
for HEMI in L R; do
    wb_command -surface-resample \
        "${MRI_DIR}/${SUBJECT}.${HEMI}.midthickness.native.surf.gii" \
        "${TMP_DIR}/${SUBJECT}.${HEMI}.sphere.fsaverage.native.surf.gii" \
        "${STANDARD_MESHES}/resample_fsaverage/fsaverage_std_sphere.${HEMI}.${TARGET_RES}k_fsavg_${HEMI}.surf.gii" \
        BARYCENTRIC \
        "${TMP_DIR}/${SUBJECT}.${HEMI}.midthickness.${TARGET_RES}k_fsavg_${HEMI}.surf.gii"
    
    if [ $? -ne 0 ]; then
        echo "Error: Step 2 failed (hemisphere: ${HEMI})"
        rm -rf ${TMP_DIR}
        exit 1
    fi
done

for RUN in "${RUNS[@]}"; do
    echo "Processing run: ${RUN}"
    INPUT_CIFTI="${FMRI_DIR}/rfMRI_${RUN}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
    # Step 3: Separate the input CIFTI file into left and right hemisphere GIFTI files (each containing all time series)
    wb_command -cifti-separate \
        "${INPUT_CIFTI}" \
        COLUMN \
        -metric CORTEX_LEFT "${TMP_DIR}/${SUBJECT}.${RUN}.L.all_timepoints.func.gii" \
        -metric CORTEX_RIGHT "${TMP_DIR}/${SUBJECT}.${RUN}.R.all_timepoints.func.gii"

    if [ $? -ne 0 ]; then
        echo "Error: Step 3 failed - separating CIFTI into GIFTI"
        rm -rf ${TMP_DIR}
        exit 1
    fi

    # Step 4: Directly resample the entire GIFTI files (with all time points) to fsaverage space
    for HEMI in L R; do
        wb_command -metric-resample \
            "${TMP_DIR}/${SUBJECT}.${RUN}.${HEMI}.all_timepoints.func.gii" \
            "${STANDARD_MESHES}/resample_fsaverage/fs_LR-deformed_to-fsaverage.${HEMI}.sphere.32k_fs_LR.surf.gii" \
            "${STANDARD_MESHES}/resample_fsaverage/fsaverage_std_sphere.${HEMI}.${TARGET_RES}k_fsavg_${HEMI}.surf.gii" \
            ADAP_BARY_AREA \
            "${OUTPUT_DIR}/${SUBJECT}.${RUN}.${HEMI}.rfMRI.fsavg_${TARGET_RES}k.func.gii" \
            -area-surfs \
            "${MRI_DIR}/${SUBJECT}.${HEMI}.midthickness_MSMAll.32k_fs_LR.surf.gii" \
            "${TMP_DIR}/${SUBJECT}.${HEMI}.midthickness.${TARGET_RES}k_fsavg_${HEMI}.surf.gii"
        
        if [ $? -ne 0 ]; then
            echo "Error: Resampling failed for hemisphere ${HEMI}"
            rm -rf ${TMP_DIR}
            exit 1
        fi
    done
done

rm -rf ${TMP_DIR}