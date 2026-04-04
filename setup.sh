#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
LTX2_DIR="$PROJECT_DIR/LTX-2"
MODELS_DIR="$PROJECT_DIR/models"
GEMMA_DIR="$MODELS_DIR/gemma_root"

echo "=== LTX-2.3 Image+Audio to Video Setup ==="
echo ""

# ---------------------------------------------------------------
# 1. Clone LTX-2 repo
# ---------------------------------------------------------------
if [ -d "$LTX2_DIR" ]; then
    echo "[OK] LTX-2 repo already cloned"
else
    echo "[*] Cloning Lightricks/LTX-2..."
    git clone https://github.com/Lightricks/LTX-2.git "$LTX2_DIR"
fi

# ---------------------------------------------------------------
# 2. Create Python venv and install packages
# ---------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    echo "[OK] Virtual environment already exists"
else
    echo "[*] Creating Python virtual environment..."
    python3.11 -m venv "$VENV_DIR" 2>/dev/null || \
    python3.12 -m venv "$VENV_DIR" 2>/dev/null || \
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[*] Upgrading pip..."
pip install --upgrade pip wheel setuptools > /dev/null

echo "[*] Installing PyTorch..."
pip install torch torchvision torchaudio > /dev/null 2>&1

echo "[*] Installing LTX-2 packages (ltx-core + ltx-pipelines)..."
pip install -e "$LTX2_DIR/packages/ltx-core" > /dev/null 2>&1
pip install -e "$LTX2_DIR/packages/ltx-pipelines" > /dev/null 2>&1

echo "[*] Installing additional dependencies..."
pip install pyyaml imageio[ffmpeg] soundfile scipy pillow > /dev/null 2>&1

# ---------------------------------------------------------------
# 3. Create models directory
# ---------------------------------------------------------------
mkdir -p "$MODELS_DIR"
mkdir -p "$GEMMA_DIR"
mkdir -p "$PROJECT_DIR/inputs"
mkdir -p "$PROJECT_DIR/outputs"

# ---------------------------------------------------------------
# 4. Verify model files
# ---------------------------------------------------------------
echo ""
echo "=== Model Verification ==="

check_model() {
    local name="$1" path="$2"
    if [ -f "$path" ]; then
        size=$(du -sh "$path" | cut -f1)
        echo "[OK] $name ($size)"
    else
        echo "[!!] MISSING: $name"
        echo "     Expected at: $path"
    fi
}

check_model "LTX-2.3 Checkpoint" "$MODELS_DIR/ltx-2.3-22b-dev.safetensors"
check_model "Distilled LoRA" "$MODELS_DIR/ltx-2.3-22b-distilled-lora-384.safetensors"
check_model "Spatial Upscaler" "$MODELS_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
check_model "Gemma Model" "$GEMMA_DIR/model.safetensors"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "If models are missing, download them:"
echo "  LTX-2.3 models:  https://huggingface.co/Lightricks/LTX-2.3"
echo "  Gemma 3 12B:      https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
echo ""
echo "Place model files in: $MODELS_DIR/"
echo "Place Gemma model + tokenizer files in: $GEMMA_DIR/"
echo ""
echo "Usage:"
echo "  source $VENV_DIR/bin/activate"
echo "  python generate.py --image inputs/photo.png --audio inputs/speech.wav --prompt 'A person speaking clearly'"
echo ""
