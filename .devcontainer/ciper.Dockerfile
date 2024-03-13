FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y tzdata

RUN apt-get install -y python3.8 python3.8-dev python3-pip curl git font-manager && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

ARG USERNAME

# Create the user
RUN groupadd $USERNAME \
    && useradd -g $USERNAME -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

RUN pip install torch==1.13.1 torchvision==0.14.1 

RUN pip install matplotlib scipy timm ptflops wandb PyYAML tqdm scikit-learn
RUN pip install natsort pandas plotly kaleido prompt_toolkit

# RUN pip install matplotlib scipy timm ptflops tensorboardX wandb

ENV SHELL /bin/bash