#!/bin/bash

# generate functional connectivity script
# this script will invoke the two scripts, fslr_to_fsaverage.sh and compute_fc.py

DATA_DIR="/mnt/f/fMRI"
OUTPUT_DIR="/mnt/d/Download"
TMP_DIR="/home/qwer/generateFC/fsaverage_func_gifti"
KEEP_ALIVE_FILE="/mnt/f/MRI/.keep_alive"
START_FROM="${1}"

fail_subjects=()
found_start=false

if [ -z "${START_FROM}" ]; then
    found_start=true
fi

keep_disk_awake() {
    while true; do
        touch "$KEEP_ALIVE_FILE"
        sleep 30
    done
}

keep_disk_awake &
KEEP_ALIVE_PID=$!

trap "kill $KEEP_ALIVE_PID 2>/dev/null; rm -f '$KEEP_ALIVE_FILE'" EXIT

for item in "${DATA_DIR}"/*; do
    
    if [ -d "${item}" ]; then
        dirname=$(basename "${item}")
        subject_id="${dirname:0:6}"

        if [ -n "${START_FROM}" ] && [ "${found_start}" = false ]; then
            if [ "${subject_id}" = "${START_FROM}" ]; then
                found_start=true
            else
                echo "Skipping ${subject_id} (before ${START_FROM})"
                continue
            fi
        fi

        mkdir -p "${TMP_DIR}"
        if ! /home/qwer/generateFC/fslr_to_fsaverage.sh "${subject_id}" 164 "${TMP_DIR}"; then
            echo "Error: fslr_to_fsaverage.sh failed for ${subject_id}"
            rm -rf "${TMP_DIR}"
            fail_subjects+=("${subject_id}")
            continue
        fi
        if ! /home/qwer/SFC/bin/python3 /home/qwer/generateFC/compute_fc.py "${subject_id}" "${OUTPUT_DIR}"; then
            echo "Error: compute_fc.py failed for ${subject_id}"
            rm -rf "${TMP_DIR}"
            fail_subjects+=("${subject_id}")
            continue
        fi
        rm -rf "${TMP_DIR}"
    fi
done

kill $KEEP_ALIVE_PID 2>/dev/null
rm -f "$KEEP_ALIVE_FILE"

echo "=== Process completed successfully! ==="

echo "Failed subjects: ${#fail_subjects[@]}"

if [ ${#fail_subjects[@]} -gt 0 ]; then
    echo "List of failed subjects:"
    for subject in "${fail_subjects[@]}"; do
        echo "  ${subject}"
    done
fi