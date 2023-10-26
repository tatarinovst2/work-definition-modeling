#!/bin/bash

pip3 install -r requirements.txt
pip3 install -r requirements_train.txt

# Install CUDA-enabled PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify the installation
python3 -c "import torch; print('PyTorch version:', torch.__version__)"

echo "Installation completed."
