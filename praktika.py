import cv2
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import segmentation_models as sm
import tqdm

from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import MeanIoU, IoU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from collections import Counter
from tensorflow.experimental.numpy import ravel
from tensorflow import stack, cast, int32, make_ndarray
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BACKBONE = 'mobilenet'

preprocess_input = sm.get_preprocessing(BACKBONE)
BATCH_SIZE = 64
SEED = 50
RAW_N_CLASSES = 16
N_CLASSES = 2

def merge_classes(mask, cls1, cls2, raw_num_classes=RAW_N_CLASSES):
    mask[mask == cls1] = cls2
    for i in range(cls1 + 1, raw_num_classes):
        mask[mask==i] = i - 1
    return mask
def merge_classes(mask, cls1, cls2, raw_num_classes=RAW_N_CLASSES):
    mask[mask == cls1] = cls2
    for i in range(cls1 + 1, raw_num_classes):
        mask[mask==i] = i - 1
    return mask

def preprocess_data(img, mask, num_class):
    img = preprocess_input(img)
    
    # merge some classes
    mask = merge_classes(mask, 15, 0)
    mask = merge_classes(mask, 14, 0)
    mask = merge_classes(mask, 13, 0)
    mask = merge_classes(mask, 12, 0)
    mask = merge_classes(mask, 11, 0)
    mask = merge_classes(mask, 10, 0)
    mask = merge_classes(mask, 8, 0)
    mask = merge_classes(mask, 7, 0)
    mask = merge_classes(mask, 6, 0)
    mask = merge_classes(mask, 5, 0)
    mask = merge_classes(mask, 4, 0)
    mask = merge_classes(mask, 3, 0)
    mask = merge_classes(mask, 2, 0)
    mask = merge_classes(mask, 1, 0)
    mask = to_categorical(mask, num_class)
    return (img, mask)

def trainGenerator(train_img_path, train_mask_path, num_class, aug=True, show_raw=False):
    
    img_data_gen_args = dict(horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')
    if aug:
        image_datagen = ImageDataGenerator(**img_data_gen_args)
        mask_datagen = ImageDataGenerator(**img_data_gen_args)
    else:
        image_datagen = ImageDataGenerator()
        mask_datagen = ImageDataGenerator()
    
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode = None,
        batch_size = BATCH_SIZE,
        seed = SEED)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = BATCH_SIZE,
        seed = SEED)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        if show_raw:
            raw_img = np.copy(img)
            img, mask = preprocess_data(img, mask, num_class)
            img = np.stack((img, raw_img), axis=-1)
            yield (img, mask)
        else:
            img, mask = preprocess_data(img, mask, num_class)
            yield (img, mask)


def show_mask_with_label(batch, labels, _min, _max, alpha=1):
    values, counts = np.unique(batch, return_counts=True)
    counts = counts / (256*256)
    im = plt.imshow(batch, cmap='viridis', vmin = _min, vmax = _max, alpha=alpha)
    colors = [im.cmap(im.norm(value)) for value in values]
    
    patches = [mpatches.Patch(color=colors[i], label=labels[values[i]]) \
               for i in range(len(values)) if counts[i] > 0.005]
    plt.legend(handles=patches)

    
kaggle_path = '/kaggle/input/satelliteimagesegmentation/'
labels = ['background','industrial land',
          'urban residential','rural residential',
          'traffic land','paddy field','irrigated land', 'dry cropland',
          'garden plot','arbor woodland',
          'shrub land','natural grassland',
          'artificial grassland','river', 'lake','pond']

train_img_path = "data/data_for_keras_aug/train_images/"
train_mask_path = "data/data_for_keras_aug/train_masks/"
train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class=N_CLASSES)

val_img_path = "data/data_for_keras_aug/val_images/"
val_mask_path = "data/data_for_keras_aug/val_masks/"
val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class=N_CLASSES)

num_train_imgs = len(os.listdir(train_img_path + "/train"))
num_val_images = len(os.listdir(val_img_path + "/val"))

steps_per_epoch = num_train_imgs // BATCH_SIZE
val_steps_per_epoch = num_val_images // BATCH_SIZE

def show_examples(generator):
    x, y = generator.__next__()
    for i in range(0,3):
        image = x[i]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.argmax(y[i], axis=2)
    return x, y

x, y = show_examples(train_img_gen)

IMG_HEIGHT = x.shape[1]
IMG_WIDTH  = x.shape[2]
IMG_CHANNELS = x.shape[3]

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = sm.Unet(BACKBONE, encoder_weights='imagenet', 
                    input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                     classes=N_CLASSES, activation='softmax')
    model.compile('Adam', loss=sm.losses.cce_jaccard_loss,
                          metrics=[sm.metrics.iou_score])
    
checkpoint_filepath = 'checkpoints/checkpoint.h5'
model.load_weights(checkpoint_filepath)

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_freq = 25,
    monitor=sm.metrics.iou_score,
    mode='max')

# model.fit(train_img_gen,
#                 steps_per_epoch=steps_per_epoch,
#                 epochs=5,
#                 verbose=1,
#                 validation_data=val_img_gen,
#                 validation_steps=val_steps_per_epoch,
#                 callbacks=[model_checkpoint_callback],
#          )

model.save_weights(f'{BACKBONE}_14classes.h5')

labels = ['background','arbor woodland']

eval_preds_img_gen = trainGenerator(val_img_path, 
                                    val_mask_path, 
                                    num_class=N_CLASSES, 
                                    show_raw=True)

test_image_batch, test_mask_batch = eval_preds_img_gen.__next__()
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3) 

test_pred_batch = model.predict(test_image_batch[:, :, :, :, 0])
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)

img_num = random.randint(0, test_image_batch.shape[0]-4)

# Папка для сохранения результатов
output_folder = 'results'
os.makedirs(output_folder, exist_ok=True)

for i in range(0, 20):
    combined_data = np.array(list(np.unique(test_mask_batch_argmax[img_num+i])) + \
                         list(np.unique(test_pred_batch_argmax[img_num+i])))
    _min, _max = np.amin(combined_data), np.amax(combined_data)
    
    plt.figure(figsize=(15, 15), dpi=80)
    
    plt.subplot(1,3,1)
    plt.imshow(test_image_batch[:, :, :, :, 1][img_num+i].astype('uint8'))
    
    plt.subplot(1,3,2)
    plt.imshow(test_image_batch[:, :, :, :, 1][img_num+i].astype('uint8'))
    plt.subplot(1,3,2)
    show_mask_with_label(test_mask_batch_argmax[img_num+i], labels, _min, _max, alpha=0.9)   
    
    plt.subplot(1,3,3)
    show_mask_with_label(test_pred_batch_argmax[img_num+i], labels, _min, _max)
    plt.show()

    # Сохранение результатов
    original_image = test_image_batch[:, :, :, :, 1][img_num+i].astype('uint8')
    mask_image = test_mask_batch_argmax[img_num+i]
    pred_image = test_pred_batch_argmax[img_num+i]

    original_path = os.path.join(output_folder, f'original_{i}.png')
    mask_path = os.path.join(output_folder, f'mask_{i}.png')
    pred_path = os.path.join(output_folder, f'predicted_{i}.png')

    plt.imsave(original_path, original_image)
    plt.imsave(mask_path, mask_image, cmap='gray')
    plt.imsave(pred_path, pred_image, cmap='gray')