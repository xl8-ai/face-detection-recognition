from flask import Flask, request
import jsonpickle
import logging
from insightface.app.face_analysis import FaceAnalysis
import numpy as np
import argparse

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

app = Flask(__name__)


fa = FaceAnalysis(det_name='retinaface_r50_v1',
                  rec_name='arcface_r100_v1',
                  ga_name='genderage_v1')

@app.route("/face-analysis", methods=["POST"])
def extract_frames():
    files = request.files['image']
    # convert bytes to string as json should be string to decode.
    files = files.read()
    files = jsonpickle.decode(files)

    app.logger.info(f"{files} received")

    frame = np.array(files)

    app.logger.info(f"extraing features ...")
    list_of_features = fa.get(frame)
    app.logger.info(f"features extracted!")

    app.logger.info(f"In total of {len(list_of_features)} faces detected!")
    
    results_frame = []
    for features in list_of_features:
        feature_dict = {'age': features.age,
                        'bbox': features.bbox,
                        'det_score': features.det_score,
                        'gender': features.gender,
                        'landmark': features.landmark,
                        'normed_embedding': features.normed_embedding
                        }
        results_frame.append(feature_dict)

    response = {'fa_results': results_frame}
    response_pickled = jsonpickle.encode(response)

    app.logger.info("json-pickle is done.")

    return response_pickled


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=-1, help='-1 means CPU')
    args = parser.parse_args()
    args = vars(args)
    fa.prepare(ctx_id=args['gpu_id'])

    app.run(host='0.0.0.0', port=10002)
