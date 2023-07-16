# docker run --rm -d -p 10002:10002 --gpus all leo/face-detection-recognition-cuda
# docker run --rm -it -p 10002:10002 -e BATCH_SIZE=6 -e REC_BATCH_SIZE=6 -e GPU_ID=0 -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda

# docker run --rm -d -p 10002:10002 -e BATCH_SIZE=24 -e REC_BATCH_SIZE=24 -e GPU_ID=0 -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda
# docker run --rm -d -p 10003:10002 -e BATCH_SIZE=24 -e REC_BATCH_SIZE=24 -e GPU_ID=1 -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda


docker run --rm -d -p 10002:10002 -e BATCH_SIZE=6 -e REC_BATCH_SIZE=6 -e GPU_ID=0 -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda
docker run --rm -d -p 10003:10002 -e BATCH_SIZE=6 -e REC_BATCH_SIZE=6 -e GPU_ID=1 -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda
docker run --rm -d -p 10004:10002 -e BATCH_SIZE=6 -e REC_BATCH_SIZE=6 -e GPU_ID=2 -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda
docker run --rm -d -p 10005:10002 -e BATCH_SIZE=6 -e REC_BATCH_SIZE=6 -e GPU_ID=3 -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda
