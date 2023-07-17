

#!/bin/bash
# author: luis arandas
# date: 07-03-2023

libs=(
    "guided-diffusion"  # clip-guidance worked out
    "AdaBins"           # depth estimation #1
    "MiDaS"             # depth estimation #2
    "pytorch3d-lite"    # three-d projection
    "ResizeRight"       # differentiable library
)

libsURL=(
    # to include depth libraries need selected forks to add path support
    # please download models separately
    # : 512x512_diffusion_uncond.pt, or 256x256
    # : AdaBins_nyu.py
    # : dpt_large_midas_2f21e586.pt
    # : secondary_imagenet.pt
    "https://github.com/crowsonkb/guided-diffusion"
    "https://github.com/luisArandas/AdaBins" 
    "https://github.com/luisArandas/MiDaS.git" 
    "https://github.com/MSFTserver/pytorch3d-lite.git"
    "https://github.com/assafshocher/ResizeRight.git"
)

echo "[installing environment]"

nvidia-smi

if conda env list | grep -q "virtual-timeline-clipguided"; then
    echo "[conda env 'virtual-timeline-clipguided' check]"
else
    echo "[conda env 'virtual-timeline-clipguided' not found]"
    conda env create -f environment.yml
fi

conda init
conda deactivate
conda activate virtual-timeline-clipguided

# check lib folder for diffusion run

if [ ! -d "libs/" ]; then
    echo "[libs/ directory not found, creating]"
    mkdir libs/
else
    echo "[libs/ directory found]"
fi

# loop through libraries and check if they exist

for i in "${libs[@]}"; do
    lib_nr="${libs[$i]}"
    if [ ! -d "libs/$lib" ]; then
        echo "[$lib directory not found, cloning]"
        if [ $i -eq 2 ]; then
            git clone -b v3 "${libsURL}" "libs/$lib"
        else
            git clone "${libsURL}" "libs/$lib"
        fi
    else
        echo "[$i directory found]"
    fi
done

# assumes user gets the pretrained models