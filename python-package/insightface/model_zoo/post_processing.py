import numpy as np


def anchors_plane(height, width, stride, base_anchors):
    """
    Parameters
    ----------
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: (A, 4) a base set of anchors
    Returns
    -------
    all_anchors: (height, width, A, 4) ndarray of anchors spreading over the plane
    """
    A = base_anchors.shape[0]
    all_anchors = np.zeros((height, width, A, 4), dtype=np.float32)
    for iw in range(width):
        sw = iw * stride
        for ih in range(height):
            sh = ih * stride
            for k in range(A):
                all_anchors[ih, iw, k, 0] = base_anchors[k, 0] + sw
                all_anchors[ih, iw, k, 1] = base_anchors[k, 1] + sh
                all_anchors[ih, iw, k, 2] = base_anchors[k, 2] + sw
                all_anchors[ih, iw, k, 3] = base_anchors[k, 3] + sh
    return all_anchors

def nms(dets, nms_threshold):
    thresh = nms_threshold
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
    
def landmark_pred(boxes, landmark_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, landmark_deltas.shape[1]))
    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    pred = landmark_deltas.copy()
    for i in range(5):
        pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
        pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
    return pred


def clip_pad(tensor, pad_shape):
    """
    Clip boxes of the pad area.
    :param tensor: [n, c, H, W]
    :param pad_shape: [h, w]
    :return: [n, c, h, w]
    """
    H, W = tensor.shape[2:]
    h, w = pad_shape

    if h < H or w < W:
        tensor = tensor[:, :, :h, :w].copy()

    return tensor


def bbox_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0:1]
    dy = box_deltas[:, 1:2]
    dw = box_deltas[:, 2:3]
    dh = box_deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if box_deltas.shape[1] > 4:
        pred_boxes[:, 4:] = box_deltas[:, 4:]

    return pred_boxes

anchor_plane_cache = {}
def post_processing_face_detection(batch_size, net_out, 
                                   use_landmarks, _feat_stride_fpn, _num_anchors, 
                                   _anchors_fpn, threshold, landmark_std, 
                                   nms_threshold, scale):
    list_det = []
    list_landmarks = []
    for k in range(batch_size):
        proposals_list = []
        scores_list = []
        landmarks_list = []
        ims = []
        
        if False:
            list_det.append(np.zeros((0, 5)))
            list_landmarks.append(np.zeros((0, 5, 2)))
            continue
            
        for _idx, s in enumerate(_feat_stride_fpn):
            _key = 'stride%s' % s
            stride = int(s)
            if use_landmarks:
                idx = _idx * 3
            else:
                idx = _idx * 2
            scores = net_out[idx]
            scores = scores[:, _num_anchors['stride%s' % s]:, :, :]
            idx += 1
            bbox_deltas = net_out[idx]

            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]
            A = _num_anchors['stride%s' % s]
            K = height * width
            key = (height, width, stride)
            if key in anchor_plane_cache:
                anchors = anchor_plane_cache[key]
            else:
                anchors_fpn = _anchors_fpn['stride%s' % s]
                anchors = anchors_plane(height, width, stride, anchors_fpn)
                anchors = anchors.reshape((K * A, 4))
                if len(anchor_plane_cache) < 100:
                    anchor_plane_cache[key] = anchors

            scores = clip_pad(scores, (height, width))
            scores2 = scores.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 1))

            bbox_deltas2 = clip_pad(bbox_deltas, (height, width))
            bbox_deltas3 = bbox_deltas2.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas3.shape[3] // A
            bbox_deltas4 = bbox_deltas3.reshape((-1, bbox_pred_len))
            bbox_deltas4 = bbox_deltas3.reshape((batch_size, -1, bbox_pred_len))

            proposals = bbox_pred(anchors, bbox_deltas4[k])
            #proposals = clip_boxes(proposals, im_info[:2])

            scores_ravel = scores2[k].ravel()
            order = np.where(scores_ravel >= threshold)[0]
            proposals = proposals[order, :]
            scores3 = scores2[k][order]

            proposals[:, 0:4] /= scale

            proposals_list.append(proposals)
            scores_list.append(scores3)

            if use_landmarks:
                idx += 1
                landmark_deltas = net_out[idx]
                landmark_deltas = clip_pad(landmark_deltas, (height, width))
                landmark_pred_len = landmark_deltas.shape[1] // A
                landmark_deltas = landmark_deltas.transpose(
                    (0, 2, 3, 1)).reshape((batch_size, -1, 5, landmark_pred_len // 5))[k]
                landmark_deltas *= landmark_std
                landmarks = landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]

                landmarks[:, :, 0:2] /= scale
                landmarks_list.append(landmarks)

        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0] == 0:
            if use_landmarks:
                landmarks = np.zeros((0, 5, 2))
            
            list_det.append(np.zeros((0, 5)))
            list_landmarks.append(landmarks)
            # return np.zeros((0, 5)), landmarks
            continue
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        proposals = proposals[order, :]
        scores = scores[order]
        if use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32,
                                                                copy=False)
        keep = nms(pre_det, nms_threshold)
        det = np.hstack((pre_det, proposals[:, 4:]))
        det = det[keep, :]
        if use_landmarks:
            landmarks = landmarks[keep]
        list_det.append(det)
        list_landmarks.append(landmarks)
        
    return list_det, list_landmarks
