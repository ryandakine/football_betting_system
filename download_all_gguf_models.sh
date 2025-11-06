#!/bin/bash

echo "🚀 Downloading all 5 GGUF models from Google Drive..."

# Ensure directory exists
mkdir -p models/gguf

# Model download mappings
declare -A models
models["12eQQ91pKzMlvXCVxn0qZ1KqGp6fbsRXh"]="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
models["13x1vxewfYFX9_llCEFTpkz4MWVs_ITzj"]="codellama-7b-instruct.Q4_K_M.gguf"  
models["1UYayQ-Zw-MTYsXB7D_jN0YKp9hHcHF6o"]="openchat-3.5-0106.Q4_K_M.gguf"
models["1aWXgZC4eV818cbCtx47OIZN2sCokwcfx"]="neural-chat-7b-v3-3.Q4_K_M.gguf"
models["1wyygwSyNpZjc8YtXRonYpniay3RcBET5"]="dolphin-2.6-mistral-7b.Q4_K_M.gguf"

# Download each model
for file_id in "${!models[@]}"; do
    filename="${models[$file_id]}"
    output_path="models/gguf/$filename"
    
    echo "📥 Downloading: $filename"
    
    if [ -f "$output_path" ]; then
        size=$(du -h "$output_path" | cut -f1)
        echo "   ✅ Already exists ($size) - skipping"
        continue
    fi
    
    gdown "https://drive.google.com/uc?id=$file_id" -O "$output_path"
    
    if [ $? -eq 0 ] && [ -f "$output_path" ]; then
        size=$(du -h "$output_path" | cut -f1) 
        echo "   ✅ Downloaded successfully ($size)"
    else
        echo "   ❌ Download failed"
    fi
    
    echo
done

echo "📊 Download Summary:"
echo "==================="
total_size=0
count=0

for file_id in "${!models[@]}"; do
    filename="${models[$file_id]}"
    output_path="models/gguf/$filename"
    
    if [ -f "$output_path" ]; then
        size=$(stat --format="%s" "$output_path" 2>/dev/null || stat -f "%z" "$output_path" 2>/dev/null)
        size_gb=$(echo "scale=1; $size / 1024 / 1024 / 1024" | bc -l)
        total_size=$(echo "$total_size + $size_gb" | bc -l)
        count=$((count + 1))
        echo "✅ $filename (${size_gb}GB)"
    else
        echo "❌ $filename - MISSING"
    fi
done

echo "==================="
echo "📊 Total: ${total_size}GB in $count/5 files"

if [ "$count" -eq 5 ]; then
    echo "🎉 All models downloaded successfully!"
    echo ""
    echo "Next steps:"
    echo "1. python test_practical_ensemble.py"
    echo "2. Integration with your football betting system"
else
    echo "⚠️ Some downloads failed. Check permissions and try again."
fi