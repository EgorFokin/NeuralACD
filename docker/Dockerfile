FROM continuumio/miniconda3:latest


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglfw3-dev \
    libgles2-mesa-dev \
    git-lfs \
    graphviz-dev \
    wget \
    unzip \
    git \
    curl \
    nano \
    zip \
    && rm -rf /var/lib/apt/lists/*

COPY docker/environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy



RUN mkdir /NeuralACD
run mkdir /NeuralACD/data
WORKDIR /NeuralACD


COPY train.ipynb /NeuralACD/train.ipynb
COPY data/plane_cache.json /NeuralACD/data/plane_cache.json
COPY data/train_data.h5 /NeuralACD/data/train_data.h5
COPY data/val_data.h5 /NeuralACD/data/val_data.h5
COPY model /NeuralACD/model
COPY train.py /NeuralACD/train.py

RUN conda init bash

EXPOSE 8888

RUN touch ~/.no_auto_tmux