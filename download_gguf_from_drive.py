#!/usr/bin/env python3
"""
Download GGUF Model from Google Drive
===================================

Helper script to download the GGUF model from your Google Drive.
"""

import os
import sys
from pathlib import Path

def download_with_gdown():
    """Download using gdown with file ID."""
    
    print("🔗 To download from Google Drive:")
    print("1. Go to your Google Drive")
    print("2. Find: OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf")  
    print("3. Right-click → 'Get link' → Set to 'Anyone with link can view'")
    print("4. Copy the link (looks like: https://drive.google.com/file/d/FILE_ID/view)")
    print("5. Extract the FILE_ID from the link")
    print()
    
    file_id = input("📝 Paste your Google Drive FILE_ID here: ").strip()
    
    if not file_id:
        print("❌ No file ID provided")
        return False
    
    # Clean up the file ID if full URL was pasted
    if "drive.google.com" in file_id:
        try:
            # Extract ID from various Google Drive URL formats
            if "/file/d/" in file_id:
                file_id = file_id.split("/file/d/")[1].split("/")[0]
            elif "id=" in file_id:
                file_id = file_id.split("id=")[1].split("&")[0]
        except:
            print("❌ Could not extract file ID from URL")
            return False
    
    output_path = "./models/gguf/OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf"
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading to: {output_path}")
    print("⏳ This may take a while (~12GB file)...")
    
    # Download command
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        import subprocess
        result = subprocess.run([
            "gdown", download_url, "-O", output_path
        ], check=True, capture_output=True, text=True)
        
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / (1024**3)  # GB
            print(f"✅ Download complete! ({file_size:.2f} GB)")
            return True
        else:
            print("❌ Download failed - file not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print("Try manually downloading from your Google Drive")
        return False
    except ImportError:
        print("❌ gdown not installed. Install it with: pip install gdown")
        return False

def check_existing_file():
    """Check if file already exists."""
    target_path = Path("./models/gguf/OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf")
    
    if target_path.exists():
        file_size = target_path.stat().st_size / (1024**3)  # GB
        print(f"✅ GGUF model already exists: {target_path} ({file_size:.2f} GB)")
        return True
    
    return False

def main():
    """Main download function."""
    print("📦 GGUF Model Download Helper")
    print("=" * 40)
    
    # Check if already downloaded
    if check_existing_file():
        response = input("\n🤔 File already exists. Re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("✅ Using existing file")
            return True
    
    # Download the file
    if download_with_gdown():
        print("\n🎉 Ready to run setup!")
        print("Next steps:")
        print("1. python setup_gguf_model.py")
        print("2. python test_gguf_integration.py")
        return True
    else:
        print("\n❌ Download failed")
        print("Manual alternative:")
        print("1. Download the file manually from Google Drive")
        print("2. Move it to: ./models/gguf/OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf")
        return False

if __name__ == "__main__":
    main()