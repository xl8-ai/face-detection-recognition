from collections import deque
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

# gender age estimaion are bad. We don't use them here.
fdr = FaceDetectionRecognition(det_name='retinaface_r50_v1',
                               rec_name=None,
                            #    rec_name='arcface_r100_v1',
                               ga_name=None)


work_buffer = deque()
result_buffer = deque()
image_size_original = None
image_size_new = None

def worker():
    global image_size_original, image_size_new
    
    while True:
        if len(work_buffer) == 0:
            time.sleep(1)
            continue
        images = work_buffer.popleft()
        
        app.logger.info(f"extraing features ... abt: {time.time()}")
        st = time.time()
        list_list_of_features = fdr.get(images)
        app.logger.info(f"features extracted! abt: {time.time()}, rt: {time.time() - st}")
        
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
        result_buffer.append(response_pickled)
        
        app.logger.info(f"prepared response! {time.time() - st}")
        # result_buffer.append("done")
        

thread = threading.Thread(target=worker)
thread.start()

@app.route("/result", methods=["GET"])
def get_result():
    ret = []
    while len(result_buffer) >0:
        ret.append(result_buffer.popleft())
    return ret

@app.route("/", methods=["POST"])
def face_detection_recognition():
    global image_size_original, image_size_new
    """Receive everything in json!!!
    """
    app.logger.debug(f"Receiving data ...")
    data = request.json
    data = jsonpickle.decode(data)

    app.logger.debug(f"decompressing image ...")
    images = []
    for image in data['images']:
        image = io.BytesIO(image)

        app.logger.debug(f"Reading a PIL image ...")
        image = Image.open(image)
        image_size_original = image.size

        app.logger.debug(f"Resizing a PIL image to 640 by 640 ...")
        image = resize_square_image(image, 640, background_color=(0, 0, 0))
        image_size_new = image.size

        app.logger.debug(f"Conveting a PIL image to a numpy array ...")
        image = np.array(image)

        if len(image.shape) != 3:
            app.logger.error(f"image shape: {image.shape} is not RGB!")
            del data, image
            response = {'face_detection_recognition': None}
            response_pickled = jsonpickle.encode(response)
            return response_pickled
        images.append(image)
    
    work_buffer.append(images)
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
    parser.add_argument('--gpu-id', type=int, default=-1, help='-1 means CPU')
    args = parser.parse_args()
    args = vars(args)
    fdr.prepare(ctx_id=args['gpu_id'])

    app.run(host='0.0.0.0', port=10002)


