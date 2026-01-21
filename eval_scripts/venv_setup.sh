REQUIREMENT_PATH="$(dirname "$0")/eval-requirements.txt"

# install python package manager uv
pip install uv

# create venv and activate it
uv venv venv-lerobot-eval --python=3.10
source venv-lerobot-eval/bin/activate

# install requirements
uv pip install -r $REQUIREMENT_PATH && \
uv pip install "numpy==1.26.4" && \
	echo "venv setup complete. Activate it with: source lerobot-eval/bin/activate"
