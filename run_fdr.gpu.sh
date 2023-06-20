docker run --rm -d -p 10002:10002 --gpus all leo/face-detection-recognition-cuda
# docker run --rm -it -p 10002:10002 -e BATCH_SIZE=1 -e DEBUG=0 -e OLD=0 -e PROFILE=0 --gpus all leo/face-detection-recognition-cuda
