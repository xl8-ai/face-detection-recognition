# Face detection and recognition

This repo is forked from https://github.com/deepinsight/insightface. We use their face detection (retinaface) and face recognition (arcface).

I used to use their gender and age estimation as well, but the accuracy is very bad so I had to drop it.

## Running a flask server.

I made a light-weight flask server that loads the modules and waits for the client calls. 

You can either run the `app.py` directly with Python3 or in a docker container.

First download the models and unzip them.

```bash
wget https://github.com/tae898/face-detection-recognition/releases/download/models/models.zip

unzip models.zip
```


### Run directly (CPU only)

1. Go to the insightface directory (i.e. `cd insightface`)

1. Install the requirements.
    ```bash
    pip3 install -r requirements.txt
    ```

1. Install the insightface python package.

    ```bash
    cd python-package && pip install . && cd ..
    ```

2. Run both apps.
    ```bash
    python3 app.py --gpu-id -1
    ```

### Run in a docker container (recommended)

- Run on CPU

  1. Go to the insightface directory (i.e. `cd insightface`)
  
  2. Build the container.
      ```bash
      docker build -t face-detection-recognition .  
      ```

  3. Run it.
      ```bash
      docker run -it --rm -p 10002:10002 face-detection-recognition
      ```

- Run on GPU

  1. Go to the insightface directory (i.e. `cd insightface`)

  2. Build the container.
      ```bash
      docker build -f Dockerfile-cuda -t face-detection-recognition-cuda .  
      ```

  3. Run it.
      ```bash
      docker run -it --rm -p 10002:10002 --gpus all face-detection-recognition-cuda
      ```

## Making a REST POST request to the flask server.

You should send an image as json. I know this is not conventional but somehow this works really good. Below is an example code.

```python
import jsonpickle
import requests
import pickle

with open('/path/to/image', 'rb') as stream:
    frame_bytestring = stream.read()
data = {'image': frame_bytestring}
data = jsonpickle.encode(data)
response = requests.post('http://127.0.0.1:10002/', json=data)
response = jsonpickle.decode(response.text)
face_detection_recognition = response['face_detection_recognition']

with open('/path/to/save/results', 'wb') as stream:
    pickle.dump(face_detection_recognition, stream)
```

`face_detection_recognition` is a list of `dict`s. The number of `dict`s correspond to the number of faces detected in the image. Every `dict` has four key-value pairs. They are:

```
bbox: bounding box (four floating point numbers).
det_score: detection confidence score (one floating point)
landmark: five facial landmarks (5 by 2 float array)
normed_embedding: face embedding (512-dimensional floating point vector)
```

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Authors

* [Taewoon Kim](https://taewoonkim.com/) 