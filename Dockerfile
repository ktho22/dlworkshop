# Start with Caffe dependencies
# This Dockerfile is ground on Kaixhin's work
FROM kaixhin/caffe-deps
MAINTAINER Tae-Ho Kim <ktho22@kaist.ac.kr>

# Move into Caffe repo
RUN cd /root/caffe && \
  mkdir build && cd build && \
  cmake .. && \
  make -j"$(nproc)" all && \
  make install

# Add to Python path
ENV PYTHONPATH=/root/caffe/python:$PYTHONPATH

# Install git, apt-add-repository and dependencies for iTorch
RUN apt-get update && apt-get install -y \
  unzip \
  wget \
  git \
  software-properties-common \
  ipython3 \
  libssl-dev \
  libzmq3-dev \
  python-zmq \
  python-pip

# Run Torch7 installation scripts
RUN git clone https://github.com/torch/distro.git /root/torch --recursive && cd /root/torch && \
  bash install-deps && ./install.sh

# Install Jupyter Notebook for iTorch
RUN pip install jupyter notebook ipywidgets
RUN /root/torch/install/bin/luarocks install itorch

RUN git clone https://github.com/ktho22/dlworkshop.git /root/dlworkshop
RUN jupyter notebook --generate-config

WORKDIR /root/dlworkshop
RUN /bin/sh /root/caffe/data/mnist/get_mnist.sh
RUN /usr/bin/python /root/caffe/scripts/download_model_binary.py /root/caffe/models/bvlc_reference_caffenet
RUN /bin/sh /root/caffe/data/ilsvrc12/get_ilsvrc_aux.sh
RUN /bin/ln /dev/null /dev/raw1394
RUN /usr/bin/wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip
RUN /usr/bin/unzip cifar10torchsmall.zip
