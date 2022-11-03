FROM nvcr.io/nvidia/pytorch:22.04-py3
RUN python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y libopenblas-dev nasm g++ gcc python3-pip cmake autoconf libtool
RUN DEBIAN_FRONTEND=noninteractive apt install -y libjemalloc2 google-perftools
#RUN git clone --branch dev_nvjpeg_decode_log --depth 1 https://github.com/Oneflow-Inc/oneflow /oneflow
ADD oneflow /oneflow
RUN python3 -m pip install -r /oneflow/dev-requirements.txt
RUN mkdir /oneflow/build
RUN cd /oneflow/build
RUN cd /oneflow/build && cmake -DCMAKE_CUDA_ARCHITECTURES="80-real" -DCUDA_STATIC=OFF -DWITH_MLIR=YES .. -C ../cmake/caches/cn/cuda.cmake
RUN cd /oneflow/build && make -j14
ENV PYTHONPATH /oneflow/python
ADD diffusers /diffusers
RUN cd /diffusers && python3 -m pip install -e .[test]
ADD transformers /transformers
RUN cd /transformers && python3 -m pip install -e .
ADD scripts /scripts
#RUN cd /scripts && python3 prefetch_models.py
