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

COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy



COPY . /NeuralACD
WORKDIR /NeuralACD

RUN conda init bash

EXPOSE 8888

RUN touch ~/.no_auto_tmux

CMD ["conda", "run", "-n", "neural_acd", "jupyter", "notebook", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]