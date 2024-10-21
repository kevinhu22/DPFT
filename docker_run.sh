#!/bin/bash

docker run \
    --name dprt \
    -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/power-pc/Master_Kevin/DPFT:/app \
    -v /media/power-pc:/data \
    --network=host \
    --shm-size=1g \
    --runtime=nvidia \
    dprt:0.0.1 bash