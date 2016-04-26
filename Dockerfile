# Start with Caffe dependencies
# This Dockerfile is ground on Kaixhin's work
FROM kaixhin/caffe-deps
MAINTAINER Tae-Ho Kim <ktho22@kaist.ac.kr>

# Move into Caffe repo
RUN cd /root/caffe && \
# Make and move into build directory
  mkdir build && cd build && \
# CMake
  cmake .. && \
# Make
  make -j"$(nproc)" all && \
  make install

# Add to Python path
ENV PYTHONPATH=/root/caffe/python:$PYTHONPATH

# Install git, apt-add-repository and dependencies for iTorch
RUN apt-get update && apt-get install -y \
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
RUN pip install jupyter
RUN pip install notebook ipywidgets
RUN /root/torch/install/bin/luarocks install itorch

RUN git clone https://github.com/ktho22/dlworkshop.git /root/dlworkshop
RUN jupyter notebook --generate-config

WORKDIR /root/dlworkshop
