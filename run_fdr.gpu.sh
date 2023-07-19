# docker run --rm -d -p 10002:10002 --gpus all leo/face-detection-recognition-cuda
# docker run --rm -it -p 10002:10002 -e BATCH_SIZE=6 -e REC_BATCH_SIZE=6 -e GPU_ID=0 \
#   -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda --gpu_id=2 --port=10002 --input_size=640

docker run --rm -d -p 10002:10002 -e BATCH_SIZE=6 -e REC_BATCH_SIZE=6 -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda --gpu_id=0 --port=10002 --input_size=640
docker run --rm -d -p 10003:10002 -e BATCH_SIZE=6 -e REC_BATCH_SIZE=6 -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda --gpu_id=1 --port=10002 --input_size=640
docker run --rm -d -p 10004:10002 -e BATCH_SIZE=6 -e REC_BATCH_SIZE=6 -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda --gpu_id=2 --port=10002 --input_size=640
docker run --rm -d -p 10005:10002 -e BATCH_SIZE=6 -e REC_BATCH_SIZE=6 -e DEBUG=0 -e OLD=0 -e PROFILE=0 -e EMPTY=0 --gpus all leo/face-detection-recognition-cuda --gpu_id=3 --port=10002 --input_size=640
