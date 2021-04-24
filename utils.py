import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision


def to_int(array):
    return [int(elt) for elt in array]

def to_relative_coordinates(coordinates, image):
    coords = coordinates[:]
    width = image.shape[1]
    height = image.shape[0]
    coords[0] /= width
    coords[2] /= width
    coords[1] /= height
    coords[3] /= height
    return coords

def to_abs_coordinates(coordinates, image):
    coords = copy.deepcopy(coordinates)
    width = image.shape[1]
    height = image.shape[0]
    coords[0] *= width
    coords[2] *= width
    coords[1] *= height
    coords[3] *= height
    return to_int(coords)


def to_xywh(bbox, image):
    abs_coordinates = to_abs_coordinates(bbox, image)
    xmin, ymin = abs_coordinates[0], abs_coordinates[1]
    w = abs_coordinates[2]-abs_coordinates[0]
    h = abs_coordinates[3]-abs_coordinates[1]
    return xmin, ymin, w, h

def show_image_bbox(image, bboxes):
    fig, ax = plt.subplots();
    ax.imshow(torchvision.utils.make_grid(image.cpu(), padding=2, normalize=True));
    for bbox, col in zip(bboxes, ['r', 'g']):
        xmin, ymin, w, h = to_xywh(list(bbox), image)
        rect = patches.Rectangle((xmin, ymin), w, h , linewidth=1, edgecolor=col, facecolor='none');
        ax.add_patch(rect);
    plt.xticks([]);
    plt.yticks([]);
    plt.show();
    
    
def bbox_to_tensor(bboxes, images):
    tensor = torch.zeros_like(images.permute(0, 2, 3, 1)[:, :, :, 1:2])
    for i, bbox in enumerate(bboxes):
        coords = to_abs_coordinates(bbox, images.permute(0, 2, 3, 1)[i])
        xmins = coords[0]
        ymins = coords[1]
        xmaxs = coords[2]
        ymaxs = coords[3]
        tensor[i, xmins:xmaxs, ymins:ymaxs, :] = 1
    return tensor.long()

def get_iou(pred_bbox, true_bbox, image):
    SMOOTH = 1e-6
    pred_masks = bbox_to_tensor(pred_bbox, image)
    true_masks = bbox_to_tensor(true_bbox, image)
    outputs = pred_masks.squeeze()  # BATCH x 1 x H x W => BATCH x H x W
    labels = true_masks.squeeze()
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    return iou.mean()
