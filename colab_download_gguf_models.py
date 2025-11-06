#!/usr/bin/env python3
"""
Google Colab Script: Download GGUF Models to Drive
=================================================

Run this in Google Colab to quickly download the 3 selected models
directly to your Google Drive. Much faster than downloading locally.

Copy and paste each cell into a new Colab notebook.
"""

print("""
🚀 COLAB CELL 1: Setup and Mount Drive
Copy this into the first cell of your Colab notebook:
""")

colab_cell_1 = '''
# Install required packages
!pip install huggingface_hub

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create models directory in Drive
!mkdir -p "/content/drive/MyDrive/gguf_models"

print("✅ Setup complete! Drive mounted and directory created.")
'''

print(colab_cell_1)

print("""
🔄 COLAB CELL 2: Download Models
Copy this into the second cell:
""")

colab_cell_2 = '''
from huggingface_hub import hf_hub_download
import os
import time

# All 5 models for maximum ensemble diversity
models_to_download = [
    {
        "name": "Mistral-7B-Instruct",
        "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", 
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_gb": 4.1
    },
    {
        "name": "CodeLlama-7B-Instruct", 
        "repo": "TheBloke/CodeLlama-7B-Instruct-GGUF",
        "filename": "codellama-7b-instruct.Q4_K_M.gguf", 
        "size_gb": 3.8
    },
    {
        "name": "OpenChat-7B",
        "repo": "TheBloke/openchat-3.5-0106-GGUF", 
        "filename": "openchat-3.5-0106.Q4_K_M.gguf",
        "size_gb": 3.9
    },
    {
        "name": "Neural-Chat-7B (Sports)",
        "repo": "TheBloke/neural-chat-7B-v3-3-GGUF",
        "filename": "neural-chat-7b-v3-3.Q4_K_M.gguf", 
        "size_gb": 3.9
    },
    {
        "name": "Dolphin-Mistral-7B (Uncensored)",
        "repo": "TheBloke/dolphin-2.6-mistral-7B-GGUF",
        "filename": "dolphin-2.6-mistral-7b.Q4_K_M.gguf",
        "size_gb": 4.0
    }
]

print(f"📦 Downloading {len(models_to_download)} models (~{sum(m['size_gb'] for m in models_to_download):.1f}GB total)")
print("This will take 5-15 minutes on Colab's fast connection...")

successful_downloads = []

for i, model in enumerate(models_to_download, 1):
    print(f"\\n🔄 [{i}/{len(models_to_download)}] Downloading {model['name']} ({model['size_gb']}GB)")
    
    target_path = f"/content/drive/MyDrive/gguf_models/{model['filename']}"
    
    # Check if already exists
    if os.path.exists(target_path):
        file_size = os.path.getsize(target_path) / (1024**3)
        print(f"   ✅ Already exists ({file_size:.1f}GB) - skipping")
        successful_downloads.append(model)
        continue
    
    try:
        start_time = time.time()
        
        # Download directly to Drive
        downloaded_path = hf_hub_download(
            repo_id=model["repo"],
            filename=model["filename"], 
            local_dir="/content/drive/MyDrive/gguf_models",
            local_dir_use_symlinks=False
        )
        
        download_time = time.time() - start_time
        actual_size = os.path.getsize(target_path) / (1024**3)
        speed_mbps = (actual_size * 1024 * 8) / download_time  # Mbps
        
        print(f"   ✅ Downloaded in {download_time:.1f}s ({speed_mbps:.0f} Mbps)")
        successful_downloads.append(model)
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")

print(f"\\n🎉 Download Summary:")
print(f"✅ Successfully downloaded: {len(successful_downloads)}/{len(models_to_download)} models")

total_size = 0
for model in successful_downloads:
    path = f"/content/drive/MyDrive/gguf_models/{model['filename']}"
    if os.path.exists(path):
        size_gb = os.path.getsize(path) / (1024**3)
        total_size += size_gb
        print(f"   📁 {model['name']}: {size_gb:.1f}GB")

print(f"💾 Total size: {total_size:.1f}GB")
print(f"📂 Location: /content/drive/MyDrive/gguf_models/")
'''

print(colab_cell_2)

print("""
📋 COLAB CELL 3: Verify Downloads 
Copy this into the third cell to verify everything:
""")

colab_cell_3 = '''
import os

models_dir = "/content/drive/MyDrive/gguf_models"
expected_files = [
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "codellama-7b-instruct.Q4_K_M.gguf",
    "openchat-3.5-0106.Q4_K_M.gguf", 
    "neural-chat-7b-v3-3.Q4_K_M.gguf",
    "dolphin-2.6-mistral-7b.Q4_K_M.gguf"
]

print("🔍 Verification Results:")
print("=" * 40)

total_size = 0
for filename in expected_files:
    filepath = os.path.join(models_dir, filename)
    if os.path.exists(filepath):
        size_gb = os.path.getsize(filepath) / (1024**3)
        total_size += size_gb
        print(f"✅ {filename} ({size_gb:.1f}GB)")
    else:
        print(f"❌ {filename} - MISSING")

print("=" * 40)
print(f"📊 Total: {total_size:.1f}GB in {len([f for f in expected_files if os.path.exists(os.path.join(models_dir, f))])}/5 files")

if total_size > 18:
    print("🎉 All models downloaded successfully!")
    print("\\nNext steps:")
    print("1. Download these files from your Google Drive to your local system") 
    print("2. Place them in: ./models/gguf/")
    print("3. Run: python test_practical_ensemble.py")
else:
    print("⚠️  Some downloads may be incomplete. Re-run cell 2.")
'''

print(colab_cell_3)

print("""
📱 INSTRUCTIONS:
==============

1. Open Google Colab (colab.research.google.com)
2. Create a new notebook
3. Copy each cell above into separate Colab cells
4. Run them in order (Cell 1 → Cell 2 → Cell 3)
5. Total download time: ~5-15 minutes on Colab
6. When done, download the files from your Drive to ./models/gguf/

The models will be in your Google Drive at:
/MyDrive/gguf_models/

Much faster than downloading locally! 🚀
""")

def create_local_download_script():
    """Also create a script to download from your Drive to local system."""
    
    download_script = '''#!/bin/bash
# Download GGUF models from Google Drive to local system
# Run this after the Colab download completes

echo "📥 Downloading GGUF models from Google Drive..."

# Ensure local directory exists
mkdir -p models/gguf

# Install gdown if needed
pip install gdown

# Download the 3 models (replace FILE_IDs with your actual Google Drive file IDs)
echo "🔄 Downloading Mistral-7B..."
gdown "FILE_ID_MISTRAL" -O models/gguf/mistral-7b-instruct-v0.2.Q4_K_M.gguf

echo "🔄 Downloading CodeLlama-7B..." 
gdown "FILE_ID_CODELLAMA" -O models/gguf/codellama-7b-instruct.Q4_K_M.gguf

echo "🔄 Downloading Dolphin-Mistral-7B..."
gdown "FILE_ID_DOLPHIN" -O models/gguf/dolphin-2.6-mistral-7b.Q4_K_M.gguf

echo "✅ Download complete! Run: python test_practical_ensemble.py"
'''
    
    return download_script

# Save the download script too
download_script = create_local_download_script()
print("\n" + "="*60)
print("📂 BONUS: Local download script created")
print("After Colab download, get the Google Drive file IDs and use this script:")
print(download_script)

if __name__ == "__main__":
    print("Copy the Colab cells above into a new Google Colab notebook!")