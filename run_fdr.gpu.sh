docker run --rm -d -p 10002:10002 --gpus all leo/face-detection-recognition-cuda
# sudo docker run --rm -it -p 10002:10002 -e BATCH_SIZE=7 -e REC_BATCH_SIZE=7 -e GPU_ID=0 -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda

