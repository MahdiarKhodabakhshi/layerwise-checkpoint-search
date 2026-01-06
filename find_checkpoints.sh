#!/bin/bash
# Quick script to check what checkpoints are available for Pythia models
# Uses curl to check HuggingFace API

echo "=" | head -c 80 && echo ""
echo "Checking Available Pythia Checkpoints"
echo "=" | head -c 80 && echo ""
echo ""

for size in 14m 70m 410m; do
    echo "Checking EleutherAI/pythia-${size}..."
    
    # Try to get model info from HuggingFace
    URL="https://huggingface.co/api/models/EleutherAI/pythia-${size}"
    
    # Check if we can access the model
    if curl -s -f "$URL" > /dev/null 2>&1; then
        echo "  ✅ Model exists"
        echo "  📝 Check manually at: https://huggingface.co/EleutherAI/pythia-${size}"
        echo "     Look for 'Files and versions' tab"
    else
        echo "  ⚠️  Could not verify (may need authentication)"
        echo "  📝 Check manually at: https://huggingface.co/EleutherAI/pythia-${size}"
    fi
    
    echo ""
done

echo "=" | head -c 80 && echo ""
echo "RECOMMENDATION:"
echo "  1. Visit each model page manually"
echo "  2. Check 'Files and versions' or 'Revisions' tab"
echo "  3. Look for: step-1000, step-2000, step-5000, step-10000, main"
echo "  4. Update configs/mteb_layersweep.yaml with verified checkpoints"
echo "=" | head -c 80 && echo ""

