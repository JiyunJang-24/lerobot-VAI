conda create -n smolvla python=3.10 -y
conda activate smolvla
pip install lerobot
pip install -e .
pip install -e ".[smolvla]"