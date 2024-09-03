#!/bin/bash
# Install gdown
pip install loguru
pip install imagekitio
pip install --upgrade --no-cache-dir gdown

# Create directories and download BFM files
mkdir -p deep_3drecon/BFM
cd deep_3drecon/BFM
gdown https://drive.google.com/uc?id=1u8A_K6uQFdkx9ipQcD24Hd_6_7BBIYDg || { echo "Download failed"; exit 1; }
gdown https://drive.google.com/uc?id=18LzQn31mqfFSLLGHi3JWLLScbFetagcs || { echo "Download failed"; exit 1; }
gdown https://drive.google.com/uc?id=1qwpobniikBED3dkO8aUuksG2sz2MoTYK || { echo "Download failed"; exit 1; }
gdown https://drive.google.com/uc?id=1PlGJduXOkzhhTeki9kaql06B9gqFlMG0 || { echo "Download failed"; exit 1; }
gdown https://drive.google.com/uc?id=1si-EzFPZbzQKibJ-BpV6pFk2f3WC1iVb || { echo "Download failed"; exit 1; }
gdown https://drive.google.com/uc?id=1wDmAdjB19-s14CxEf-jblxkAEHcCTDnB || { echo "Download failed"; exit 1; }
gdown https://drive.google.com/uc?id=1zQAhaUsN5l7yj6qDfnHf5tyjEkQCgrVq || { echo "Download failed"; exit 1; }
gdown https://drive.google.com/uc?id=1vqyTZWGoaXgin5CR8KuO1ATAP-Z-C0sH || { echo "Download failed"; exit 1; }
cd ../..

# Create checkpoints directory and download checkpoint files
mkdir -p checkpoints
cd checkpoints
gdown https://drive.google.com/uc?id=1zUNhAFS3S9PK5bKBzSXAs-r2-CjJZark || { echo "Download failed"; exit 1; }
gdown https://drive.google.com/uc?id=1-VLuesgKUHGpaj9pdA9-x1ouMrsjuQu3 || { echo "Download failed"; exit 1; }

# Unzip downloaded files
python3 -c "
import zipfile

def unzip_file(zip_path, extract_to='.'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

unzip_file('240210_real3dportrait_orig.zip', '.')
unzip_file('pretrained_ckpts.zip', '.')
"

# List contents of the directory
ls

# Return to the original directory
cd ..

# Install cmake
conda install -y cmake

# Install additional dependencies
conda install -c conda-forge dlib
conda install -y conda-forge::ffmpeg
conda install -y pyaudio
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y pytorch3d::pytorch3d
pip install cython
pip install chardet
pip install openmim==0.3.9
mim install mmcv==2.1.0
pip install -r docs/prepare_env/requirements.txt -v
pip install -r docs/prepare_env/requirements.txt -v --use-deprecated=legacy-resolver

# Install dlib with conda-forge to avoid build issues

