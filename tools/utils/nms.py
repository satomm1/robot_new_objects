import numpy as np

def remove_duplicate_boxes(correct_boxes, candidate_boxes, correct_classes, candidate_classes, iou_threshold=0.5, higher_iou_threshold=0.5):
    keep = []
    keep_indx = []
    indx = -1
    for candidate, candidate_class in zip(candidate_boxes, candidate_classes):
        indx += 1
        is_duplicate = False
        for correct, correct_class in zip(correct_boxes, correct_classes):
            iou = compute_iou(candidate, correct)
            if correct_class == candidate_class:
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            else:
                if iou > higher_iou_threshold:
                    is_duplicate = True
                    break
        if not is_duplicate:
            keep.append(candidate)
            keep_indx.append(indx)

    return keep, keep_indx

# Check for overlap between two boxes when there is no label for the candidate boxes
def remove_duplicate_boxes_no_label(correct_boxes, candidate_boxes, iou_threshold=0.6):
    keep = []
    keep_indx = []
    indx = -1
    for candidate in candidate_boxes:
        indx += 1
        is_duplicate = False
        for correct in correct_boxes:
            iou = compute_iou(candidate, correct)
            
            if iou > iou_threshold:
                is_duplicate = True
                break

            # # Compute if 90% or more of the candidate box is inside the correct box
            # if compute_intersection(candidate, correct) > 0.7:
            #     is_duplicate = True
            #     break 

            
        if not is_duplicate:
            keep.append(candidate)
            keep_indx.append(indx)

    return keep, keep_indx

def remove_overlap(boxes, iou_threshold=0.9):
    keep = []
    keep_indx = []
    indx = -1
    for i, box in enumerate(boxes):
        indx += 1
        is_duplicate = False
        for j, box2 in enumerate(boxes):
            if i == j:
                continue

            iou = compute_iou(box, box2)
            if iou > iou_threshold:
                is_duplicate = True
                break

            intersect1 = compute_intersection(box, box2)
            if intersect1 > 0.99:
                is_duplicate = True
                break
            # intersect2 = compute_intersection(box2, box)
            # if intersect1 > 0.25 and intersect2 > 0.25 and (intersect1 > 0.8 or intersect2 > 0.8):
            #     is_duplicate = True
            #     break
        if not is_duplicate:
            keep.append(box)
            keep_indx.append(indx)

    return keep, keep_indx

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def compute_intersection(box1, box2):
    # Computes the intersection between two boxes divided by the area of box1
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    return inter_area / float(box1_area)


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    assert len(boxes) == len(scores)
    if len(boxes) == 0:
        return []

    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Get coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort by score
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

