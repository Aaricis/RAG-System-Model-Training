#!/bin/bash
set -e

# === Google Drive 資料夾 ID ===
FOLDER_ID="1ZC5DRptpXbIvivdfhr2RE0EEzCxUvnB2"

# === 輸出路徑 ===
OUT_DIR="./models"
mkdir -p $OUT_DIR

# === 下載整個資料夾 ===
echo "Downloading models data from Google Drive..."
gdown --folder https://drive.google.com/drive/folders/$FOLDER_ID -O $OUT_DIR

echo "Download finished. Files saved to $OUT_DIR"