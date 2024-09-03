#!/bin/bash

# This script installs Anaconda on a Linux server.

# Step 1: Update the system package list
echo "Updating system package list..."
sudo apt-get update -y

# Step 2: Install the required dependencies for GUI packages (Debian/Ubuntu-based)
echo "Installing required dependencies..."
sudo apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

# Step 3: Download the Anaconda installer
INSTALLER_VERSION="2023.09-0"  # Replace with the desired version
ARCHITECTURE="x86_64"  # Replace with your architecture if different
INSTALLER="Anaconda3-${INSTALLER_VERSION}-Linux-${ARCHITECTURE}.sh"

echo "Downloading Anaconda installer..."
curl -O https://repo.anaconda.com/archive/${INSTALLER}

# Step 4: Verify the installer (optional)
echo "Verifying installer integrity..."
sha256sum ${INSTALLER}

# Step 5: Run the installer
echo "Running Anaconda installer..."
bash ${INSTALLER} -b -p $HOME/anaconda3

# Step 6: Initialize Anaconda
echo "Initializing Anaconda..."
$HOME/anaconda3/bin/conda init

# Step 7: Refresh the terminal
echo "Refreshing terminal..."
source ~/.bashrc

# Step 8: Verify the installation
echo "Verifying Anaconda installation..."
conda --version

echo "Anaconda installation complete!"
