import threading
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
lock = threading.Lock()

# gender age estimaion are bad. We don't use them here.
fdr = FaceDetectionRecognition(det_name='retinaface_r50_v1',
                               rec_name='arcface_r100_v1',
                               ga_name=None)


@app.route("/", methods=["POST"])
def face_detection_recognition():
    """Receive everything in json!!!
    """
    app.logger.debug(f"Receiving data ...")
    data = request.json
    data = jsonpickle.decode(data)

    app.logger.debug(f"decompressing image ...")
    image = data['image']
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

    app.logger.info(f"extraing features ...")
    app.logger.info(f"acquire lock...")
    with lock:
        list_of_features = fdr.get(image)
    
    app.logger.info(f"features extracted!")

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

    response = {'face_detection_recognition': results_frame}
    app.logger.info("json-pickle is done.")

    response_pickled = jsonpickle.encode(response)

    return response_pickled

@app.route("/shutdown", methods=["GET"])
def shutdown():
    from flask import jsonify
    import os
    import signal

    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({ "success": True, "message": "Server is shutting down..." })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=-1, help='-1 means CPU')
    args = parser.parse_args()
    args = vars(args)
    fdr.prepare(ctx_id=args['gpu_id'])

    app.run(host='0.0.0.0', port=10002)


