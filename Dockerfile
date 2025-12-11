FROM osrf/ros:humble-desktop-full

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install --upgrade "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY . /ros2_ws/src/hil-serl

WORKDIR ./ros2_ws/src/hil-serl/serl_launcher

RUN pip install -e .
RUN pip install -r requirements.txt