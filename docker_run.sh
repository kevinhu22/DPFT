docker run \
    --name dprt \
    -it \
    --gpus all \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/power-pc/Master_Kevin/DPFT:/app \
    -v /media/power-pc:/data \
    --network=host \
    --shm-size=1g \
    dprt:0.0.1 bash

