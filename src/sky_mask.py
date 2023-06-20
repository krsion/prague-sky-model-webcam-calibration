import os
import numpy as np
from collections import namedtuple
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import skimage.morphology as morphology

SIZE = (1600, 1200)

Category = namedtuple('Category', ['id', 'name', 'color'])

categories = [
    Category(0, 'road', (0, 0, 0)),
    Category(1, 'sidewalk', (0, 0, 0)),
    Category(2, 'building', (0, 0, 0)),
    Category(3, 'wall', (0, 0, 0)),
    Category(4, 'fence', (0, 0, 0)),
    Category(5, 'pole', (0, 0, 0)),
    Category(6, 'traffic light', (0, 0, 0)),
    Category(7, 'traffic sign', (0, 0, 0)),
    Category(8, 'vegetation', (0, 0, 0)),
    Category(9, 'terrain', (0, 0, 0)),
    Category(10, 'sky', (255, 255, 255)),
    Category(11, 'person', (0, 0, 0)),
    Category(12, 'rider', (0, 0, 0)),
    Category(13, 'car', (0, 0, 0)),
    Category(14, 'truck', (0, 0, 0)),
    Category(15, 'bus', (0, 0, 0)),
    Category(16, 'train', (0, 0, 0)),
    Category(17, 'motorcycle', (0, 0, 0)),
    Category(18, 'bicycle', (0, 0, 0)),
]
colors_matrix = np.array([x.color for x in categories])
name2color = {x.name: x.color for x in categories}


def load_folder(path, image_size):
    images = []
    for img_name in os.listdir(path):
        if not img_name.endswith('.jpg'):
            continue
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path).resize(image_size)
        images.append(np.asarray(img))
    return images


x = load_folder('data/sky2', SIZE)
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    size=SIZE[::-1])
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")


def predict(image):
    inputs = feature_extractor(images=[image], return_tensors='pt')
    logits = None
    with torch.no_grad():
        logits = model(**inputs).logits
        logits = F.interpolate(logits, scale_factor=(4, 4), mode='nearest')
        logits = logits.detach().numpy()

    segmentation = np.argmax(logits, axis=1)
    segmentation = colors_matrix[segmentation]
    return segmentation[0]


'''
for index, filename in enumerate(os.listdir('mask')):
    segmentation = predict(x[index]).astype(np.uint8)[:, :, 0]
    Image.fromarray(segmentation).save(os.path.join(result_folder, filename))
'''


def generate_skymasks():
    def predict(image):
        inputs = feature_extractor(images=[image], return_tensors='pt')
        logits = None
        with torch.no_grad():
            logits = model(**inputs).logits
            logits = F.interpolate(logits, scale_factor=(4, 4), mode='nearest')
            logits = logits.detach().numpy()

        segmentation = np.argmax(logits, axis=1)
        segmentation = colors_matrix[segmentation]
        return segmentation[0]

    x = load_folder('data/sky2', SIZE)
    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        size=SIZE[::-1])
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

    result_folder = 'data/mask2'

    for index, filename in enumerate(os.listdir('data/sky2')):
        print(filename)
        segmentation = predict(x[index]).astype(np.uint8)[:, :, 0]
        Image.fromarray(segmentation).save(
            os.path.join(result_folder, filename))


def morpho():
    l = len(os.listdir('data/mask2'))
    for index, filename in enumerate(os.listdir('data/mask2')):
        print(index, '/', l, filename)
        mask = plt.imread('data/mask2/'+filename).astype(np.uint8)//255

        position = (1, 800)
        if (mask[position]) != 1:
            print(filename, "ERROR")
        disk = morphology.disk(5)
        mask1 = morphology.flood(mask, (1, 800))
        mask2 = morphology.opening(mask1, disk)
        Image.fromarray(mask2).save('data/maskmorpho2/'+filename)

    # lysa_hora2, lysa_hora3 jsou bad


morpho()
