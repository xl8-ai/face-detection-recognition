from __future__ import division
import collections
import pdb
import mxnet as mx
import numpy as np
from numpy.linalg import norm
import mxnet.ndarray as nd
from ..model_zoo import model_zoo
from ..utils import face_align

__all__ = ['FaceAnalysis', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'landmark', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class FaceAnalysis:
    def __init__(self,
                 det_name='retinaface_r50_v1',
                 rec_name='arcface_r100_v1',
                 ga_name='genderage_v1'):
        assert det_name is not None
        self.det_model = model_zoo.get_model(det_name)
        if rec_name is not None:
            self.rec_model = model_zoo.get_model(rec_name)
        else:
            self.rec_model = None
        if ga_name is not None:
            self.ga_model = model_zoo.get_model(ga_name)
        else:
            self.ga_model = None

    def prepare(self, ctx_id, nms=0.4):
        self.det_model.prepare(ctx_id, nms)
        if self.rec_model is not None:
            self.rec_model.prepare(ctx_id)
        if self.ga_model is not None:
            self.ga_model.prepare(ctx_id)

    def get(self, imgs, det_thresh=0.8, det_scale=1.0, max_num=0):
        list_bboxes, list_landmarks = self.det_model.detect(imgs,
                                                  threshold=det_thresh,
                                                  scale=det_scale)
        
        list_ret = []
        fimg_face = []
        for img, bboxes, landmarks in zip(imgs, list_bboxes, list_landmarks):
            if bboxes.shape[0] == 0:
                list_ret.append([])
                continue
            if max_num > 0 and bboxes.shape[0] > max_num:
                area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] -
                                                        bboxes[:, 1])
                img_center = img.shape[0] // 2, img.shape[1] // 2
                offsets = np.vstack([
                    (bboxes[:, 0] + bboxes[:, 2]) / 2 - img_center[1],
                    (bboxes[:, 1] + bboxes[:, 3]) / 2 - img_center[0]
                ])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
                bindex = np.argsort(
                    values)[::-1]  # some extra weight on the centering
                bindex = bindex[0:max_num]
                bboxes = bboxes[bindex, :]
                landmarks = landmarks[bindex, :]
            ret = []
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, 0:4]
                det_score = bboxes[i, 4]
                landmark = landmarks[i]
                _img = face_align.norm_crop(img, landmark=landmark)
                embedding = None
                embedding_norm = None
                normed_embedding = None
                gender = None
                age = None
                face = Face(bbox=bbox,
                            landmark=landmark,
                            det_score=det_score,
                            embedding=embedding,
                            gender=gender,
                            age=age,
                            normed_embedding=normed_embedding,
                            embedding_norm=embedding_norm)
                fimg_face.append((_img, face))
                ret.append(face)
            list_ret.append(ret)
            
        face_replace = {}
            
        if self.rec_model is not None and fimg_face:
            # pdb.set_trace()
            embeddings = self.rec_model.get_embedding([x[0] for x in fimg_face])
            for embedding, (_, face) in zip(embeddings, fimg_face):
                embedding_norm = norm(embedding)
                normed_embedding = embedding / embedding_norm
                face_replace[id(face)] = face._replace(embedding=embedding, normed_embedding=normed_embedding, embedding_norm=embedding_norm)
                
            list_ret_new = []
            for list1 in list_ret:
                list_ret_new.append([])
                for x in list1:
                    list_ret_new[-1].append(face_replace[id(x)])
            list_ret = list_ret_new
                
        return list_ret
