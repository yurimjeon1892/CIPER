FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y python3.8 python3.8-dev python3-pip curl git && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    pip install virtualenv

RUN virtualenv -p python3.8 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install torch==1.13.1 torchvision==0.14.1 

RUN pip install matplotlib scipy timm ptflops wandb PyYAML tqdm

ENV SHELL /bin/bash