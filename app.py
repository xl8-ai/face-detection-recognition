from collections import deque
from hashlib import md5
import os
import sys
import threading
import time
from flask import Flask, request
import jsonpickle
import logging
from insightface.app.face_analysis import FaceAnalysis as FaceDetectionRecognition
import numpy as np
import argparse
import io
from PIL import Image
from utils import resize_square_image, get_original_bbox, get_original_lm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

app = Flask(__name__)
INPUT_SIZE = None
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
PARALLEL_SIZE = int(os.environ.get("PARALLEL_SIZE", 1))
IDX_WORKER = int(os.environ.get("IDX_WORKER", 0))


# gender age estimaion are bad. We don't use them here.
fdr = FaceDetectionRecognition(det_name='retinaface_r50_v1',
                            #    rec_name=None,
                               rec_name='arcface_r100_v1',
                               ga_name=None)


work_buffer = deque()
result_buffer = deque()
image_size_original = {}
image_size_new = {}
image_hash = {}
next_req_id_enqueue = BATCH_SIZE*IDX_WORKER

@app.route("/reset", methods=["POST"])
def reset():
    global next_req_id_enqueue
    assert not work_buffer
    assert not result_buffer
    assert not image_size_original
    assert not image_size_new
    assert not image_hash

    next_req_id_enqueue = BATCH_SIZE*IDX_WORKER
    return {
        "result": 0,
    }

def worker():
    global image_size_original, image_size_new, image_hash
    
    while True:
        if len(work_buffer) == 0:
            time.sleep(0.001)
            continue
        images, req_ids = work_buffer.popleft()
        
        app.logger.info(f"extraing features ... abt: {time.time()}")
        st = time.time()
        list_list_of_features = fdr.get(images)
        app.logger.info(f"features extracted! abt: {time.time()}, rt: {time.time() - st}")
        
        list_results_frame = []
        image_size_original_ = image_size_original[id(images)]
        image_size_new_ = image_size_new[id(images)] 
        image_hash_ = image_hash[id(images)]
        
    
        assert len(list_list_of_features) == len(image_size_original_)
        for idx_batch, list_of_features in enumerate(list_list_of_features):
            app.logger.info(f"In total of {len(list_of_features)} faces detected!")

            results_frame = []
            for features in list_of_features:
                bbox = get_original_bbox(features.bbox, image_size_original_[idx_batch], image_size_new_[idx_batch])
                landmark = get_original_lm(features.landmark, image_size_original_[idx_batch], image_size_new_[idx_batch])
                feature_dict = {'bbox': bbox,
                                'det_score': features.det_score,
                                'landmark': landmark,
                                'normed_embedding': features.normed_embedding,
                                'org_size': image_size_original_[idx_batch],
                                'new_size': image_size_new_[idx_batch],
                                'bbox_model': features.bbox,
                                'image_hash': image_hash_[idx_batch],
                                }
                results_frame.append(feature_dict)
            list_results_frame.append(results_frame)

        response = {'face_detection_recognition': list_results_frame}
        app.logger.info("json-pickle is done.")
        
        response_pickled = jsonpickle.encode(response)
        result_buffer.append((response_pickled, req_ids))
        
        app.logger.info(f"prepared response! {time.time() - st}")
        
        image_size_original.pop(id(images))
        image_size_new.pop(id(images))
        image_hash.pop(id(images))
        
        # result_buffer.append("done")
        

thread = threading.Thread(target=worker)
thread.start()

@app.route("/result", methods=["GET"])
def get_result():
    ret = []
    while len(result_buffer) >0:
        batch_result, req_ids = result_buffer.popleft()
        ret.append((batch_result, req_ids))
    return {
        "result": ret,
        "count_work_buffer": len(work_buffer),
    }

@app.route("/", methods=["POST"])
def face_detection_recognition():
    global image_size_original, image_size_new, image_hash, next_req_id_enqueue
    """Receive everything in json!!!
    """
    app.logger.debug(f"Receiving data ...")
    data = request.json
    data = jsonpickle.decode(data)

    app.logger.debug(f"decompressing image ...")
    images = []
    
    image_size_original_ = []
    image_size_new_ = []
    image_hash_ = []
    
    req_ids = data['req_ids']
    
    for image in data['images']:
        image_hash_code = md5(image).hexdigest()
        image = io.BytesIO(image)

        app.logger.debug(f"Reading a PIL image ...")
        image = Image.open(image)
        image_size_original_.append(image.size)

        app.logger.debug(f"Resizing a PIL image to {INPUT_SIZE} by {INPUT_SIZE} ...")
        image = resize_square_image(image, INPUT_SIZE, background_color=(0, 0, 0))
        image_size_new_.append(image.size)
        image_hash_.append(image_hash_code)

        app.logger.debug(f"Conveting a PIL image to a numpy array ...")
        image = np.array(image)

        if len(image.shape) != 3:
            app.logger.error(f"image shape: {image.shape} is not RGB!")
            del data, image
            response = {'face_detection_recognition': None}
            response_pickled = jsonpickle.encode(response)
            return response_pickled
        images.append(image)
    
    image_size_original[id(images)] = image_size_original_
    image_size_new[id(images)] = image_size_new_
    image_hash[id(images)] = image_hash_
    
    while True:
        if not next_req_id_enqueue == req_ids[0]:
            time.sleep(0.001)
            # time.sleep(0.1)
            # print("waiting", next_req_id_enqueue, req_ids[0])
            continue
        work_buffer.append((images, req_ids))
        next_req_id_enqueue = req_ids[-1]+1+(BATCH_SIZE*(PARALLEL_SIZE-1))
        break
    return "done"

    app.logger.info(f"extraing features ...")
    list_list_of_features = fdr.get(images)
    app.logger.info(f"features extracted!")
    
    list_results_frame = []
    for list_of_features in list_list_of_features:
        app.logger.info(f"In total of {len(list_of_features)} faces detected!")

        results_frame = []
        for features in list_of_features:
            bbox = get_original_bbox(features.bbox, image_size_original, image_size_new)
            landmark = get_original_lm(features.landmark, image_size_original, image_size_new)
            feature_dict = {'bbox': bbox,
                            'det_score': features.det_score,
                            'landmark': landmark,
                            'normed_embedding': features.normed_embedding
                            }
            results_frame.append(feature_dict)
        list_results_frame.append(results_frame)

    response = {'face_detection_recognition': list_results_frame}
    app.logger.info("json-pickle is done.")

    response_pickled = jsonpickle.encode(response)

    return response_pickled


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='-1 means CPU')
    parser.add_argument('--port', type=int, default=10002)
    parser.add_argument('--input_size', type=int, default=640)
    args = parser.parse_args()
    args = vars(args)
    
    INPUT_SIZE = args['input_size']

    fdr.prepare(ctx_id=args['gpu_id'])
    app.run(host='0.0.0.0', port=args['port'])
