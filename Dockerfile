FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Fix nvidia-key error issue (NO_PUBKEY A4B469963BF863CC)
RUN rm /etc/apt/sources.list.d/*.list

# Installing apt packages
RUN export DEBIAN_FRONTEND=noninteractive 	&& apt -y update --no-install-recommends 	&& apt -y install --no-install-recommends 	  git wget tmux vim zsh build-essential cmake ninja-build libopenblas-dev libsparsehash-dev 	&& apt autoremove -y 	&& apt clean -y 	&& export DEBIAN_FRONTEND=dialog

# Install Pointcept environment
RUN conda install h5py pyyaml -c anaconda -y
RUN conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
RUN conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y

RUN pip install --upgrade pip
RUN pip install torch-geometric
RUN pip install spconv-cu117
RUN pip install open3d

## Build MinkowskiEngine
# RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git
# WORKDIR /workspace/MinkowskiEngine
# RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0" python setup.py install --blas=openblas --force_cuda
WORKDIR /workspace

# Build pointops
RUN git clone https://github.com/Pointcept/Pointcept.git
RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0" pip install Pointcept/libs/pointops -v

# Build pointgroup_ops
RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0" pip install Pointcept/libs/pointgroup_ops -v

# Build swin3d
RUN TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.0" pip install -U git+https://github.com/microsoft/Swin3D.git -v

# Added to remove error for train (AssertionError: Trainer: DBBD: PointTransformerV3: Make sure flash_attn is installed.)
RUN pip install flash-attn
RUN pip install debugpy
RUN apt-get update && apt-get install -y libx11-6
RUN apt-get update && apt-get install -y libgl1-mesa-glx libgl1-mesa-dri
