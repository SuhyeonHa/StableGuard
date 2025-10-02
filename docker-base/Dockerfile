# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
# FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION "python"


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN  apt-get update -y && \
     apt-get upgrade -y && \
     apt-get dist-upgrade -y && \
     apt-get -y autoremove && \
     apt-get clean

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    zip \
    rsync \
    vim \
    git \
    libgl1-mesa-glx \
    libglib2.0-0

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD [ "/bin/bash" ]