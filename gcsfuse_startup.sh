#!/bin/bash

# This script will be executed on Ray worker nodes upon startup to mount GCS buckets.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting GCS FUSE mount setup..."

# Install gcsfuse (if not already present). This varies by OS.
# For Debian/Ubuntu based images (common for GCP VMs):
if ! command -v gcsfuse &> /dev/null
then
    echo "gcsfuse not found, installing..."
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
    echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install -y gcsfuse
    echo "gcsfuse installed."
else
    echo "gcsfuse already installed."
fi

# Create mount points (these should match what's in your config.yaml's gcs_fuse_mount_base)
MOUNT_BASE="/mnt/gcs"
sudo mkdir -p "${MOUNT_BASE}"
echo "Created mount base directory: ${MOUNT_BASE}"

# --- Define your GCS buckets to mount ---
GCS_BUCKETS_TO_MOUNT=(
    "rsabawi-hpc-sandbox-926388-genomics-reads"
    "rsabawi-hpc-sandbox-926388-genomics-refs"
    "rsabawi-hpc-sandbox-926388-staging-area"
    "rsabawi-hpc-sandbox-926388-genomics-results"
    "rsabawi-hpc-sandbox-926388-tensorboard-logs"
)

# Mount each bucket
for bucket in "${GCS_BUCKETS_TO_MOUNT[@]}"; do
    MOUNT_PATH="${MOUNT_BASE}/${bucket}"
    sudo mkdir -p "${MOUNT_PATH}"
    echo "Attempting to mount gs://${bucket} to ${MOUNT_PATH}..."
    gcsfuse --implicit-dirs "${bucket}" "${MOUNT_PATH}" &> /var/log/gcsfuse_${bucket}.log &
    sleep 2 # Small delay for mount process to start
    if mountpoint -q "${MOUNT_PATH}"; then
        echo "Successfully mounted gs://${bucket} to ${MOUNT_PATH}"
    else
        echo "Failed to mount gs://${bucket} to ${MOUNT_PATH}. Check /var/log/gcsfuse_${bucket}.log for errors."
    fi
done

echo "GCS FUSE mount setup complete."