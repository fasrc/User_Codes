# alphapose

[alphapose](https://github.com/MVIG-SJTU/AlphaPose) is an accurate multi-person pose estimator. The steps below show how to install and run alphapose on fasrc. This tutorial is partially inspired from this [source](https://github.com/MVIG-SJTU/AlphaPose/issues/1206). 


### Step 1: Clean the environment to avoid cache issues
```bash
conda clean --all
rm -rfv ~/.conda/pkgs
rm -rfv ~/.cache/pip
```


### Step 2: Request an interactive job and set up the GPU device
```bash
salloc --partition gpu_test --time 04:10:00 -c 8 --mem-per-cpu 4G --gres=gpu:1
module purge
module load python
mamba clean --locks
export PYTHONNOUSERSITE=yes
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -L | awk '/MIG/ {gsub(/[()]/,"");print $NF}')
```


### Step 3: Create a conda virtual environment and install the libraries
⚠️ you might need to update the path below to use pip from your conda environment ⚠️
```bash
conda create -n alphapose python=3.10 -y
conda activate alphapose
module load cuda/11.3.1-fasrc01
conda install pip
/n/home00/$USER/.conda/envs/alphapose/bin/pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
/n/home00/$USER/.conda/envs/alphapose/bin/pip install cython matplotlib cython_bbox numpy scipy easydict opencv-python pycocotools pyyaml tensorboardx terminaltables tqdm visdom gdown
```


### Step 4: Install HalpeCOCOAPI ([source](https://github.com/MVIG-SJTU/AlphaPose/issues/1002))
```bash
module load gcc/9.5.0-fasrc01
cd ~
git clone https://github.com/HaoyiZhu/HalpeCOCOAPI.git
cd HalpeCOCOAPI/PythonAPI
python3 setup.py build develop --user
```

### Step 5: clone alpha pose and install it
```bash
cd ~
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose
python3 setup.py build develop --user
```


### Step 6: download weights and models
```bash
gdown https://drive.google.com/uc?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC
mkdir AlphaPose/detector/yolo/data & mv yolov3-spp.weights AlphaPose/detector/yolo/data/data/yolov3-spp.weights
gdown https://drive.google.com/uc?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn
mv fast_res50_256x192.pth AlphaPose/pretrained_models
```


### Step 7: test the model
```bash
  python3 scripts/demo_inference.py \
  --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
  --checkpoint pretrained_models/fast_res50_256x192.pth \
  --video test.mp4 \
  --outdir output
```


## Troubleshooting
- if you you get "ModuleNotFoundError: No module named 'detector'" after running demo_inference, add "sys.path.append('.')" to demo_inference.py after "import sys" ([source](https://github.com/MVIG-SJTU/AlphaPose/issues/1170]))
