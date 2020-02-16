FROM tensorflow/tensorflow:1.5.0-gpu-py3

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libav-tools \
    python-software-properties \
    software-properties-common \
    wget
    
RUN git clone https://github.com/pascalxia/keras.git
RUN cd keras && git checkout 6750e1e51fb106d6595f212bd7a1fd1e95e0e91e && python3 setup.py install
    
RUN pip3 install \
    feather-format \
    ffprobe \
    moviepy \
    opencv-python==3.2.0.8 \
    pandas \
    scikit-image \
    tqdm
    
RUN pip3 install \
    requests