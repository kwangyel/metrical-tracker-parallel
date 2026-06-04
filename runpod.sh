git clone https://github.com/kwangyel/metrical-tracker-parallel.git
mv metrical-tracker-parallel metrical-tracker
cd metrical-tracker

pip install --upgrade pip setuptools wheel ninja
pip install yacs loguru tqdm trimesh scipy scikit-image matplotlib tensorboard \
  face-alignment==1.3.5 \
  opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84 \
  "protobuf==3.20.3" "mediapipe==0.10.9" \
  imageio networkx numba pillow pyyaml

pip install -U iopath fvcore

export CUDA_HOME=/usr/local/cuda-12.4
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# RunPod PyTorch image ships numpy 2.x; opencv 4.7 wheels are numpy-1-only
pip install --force-reinstall opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84

# Re-pin mediapipe after other deps; new mediapipe (>=0.10.31) removes mp.solutions API
pip install --force-reinstall "protobuf==3.20.3" "mediapipe==0.10.9"

# FLAME pickle loading needs chumpy patched for Python 3.11 (inspect.getargspec removed)
pip uninstall -y chumpy 2>/dev/null || true
pip install --no-build-isolation \
  "git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17"

pip install gdown
apt-get update
apt-get install unzip vim wget

# flames
mkdir -p data/FLAME2020/

gdown https://drive.google.com/file/d/1455XCYSZWmfILCYKS8IPBwKJaIF-zsri/view?usp=sharing -O FLAME2020.zip
unzip -q FLAME2020.zip -d data/ && rm -rf FLAME2020.zip
mv data/FLAME2020/Readme.pdf data/FLAME2020/Readme_FLAME.pdf

gdown https://drive.google.com/file/d/1PbPtZVdBjRw5snz2Y7UWRsJnZPRWPl9C/view?usp=sharing -O TextureSpace.zip
unzip -q TextureSpace.zip -d data/FLAME2020/ && rm -rf TextureSpace.zip

gdown https://drive.google.com/file/d/1c9GPL7K7vgVDEHhOxXZrr08P4KP4Ef4R/view?usp=sharing -O FLAME_masks.zip
unzip -q FLAME_masks.zip -d data/FLAME2020/ && rm -rf FLAME_masks.zip

wget -O mesh.zip "https://keeper.mpdl.mpg.de/f/f158a430ef754edba5ec/?dl=1"
unzip -q mesh.zip -d data/ && mv data/mesh/* data/ && rm -rf data/mesh mesh.zip


mkdir -p processed_data
gdown https://drive.google.com/file/d/15Mo5t41TGI4vknGBotjKeM4gWhwgu82y/view?usp=sharing -O processed_data.tar.gz
tar -xzf processed_data.tar.gz -C processed_data

#!/bin/bash

# Ensure the target directory exists

# Write the configuration file
cat << 'EOF' > ./configs/actors/my_actor.yml
actor: './input/my_actor'
save_folder: './output/'
optimize_shape: true
optimize_jaw: true
begin_frames: 1
keyframes: [ 0, 1 ]
EOF



python scripts/colab_parallel_tracker.py \
  --cfg ./configs/actors/my_actor.yml \
  --save_folder ./output_parallel/ \
  --start_frame 3200 \
  --batch_frames 150 \
  --overlap_frames 20 \
  --num_workers 2 \
  --skip_preprocess