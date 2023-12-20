import tkinter as tk
from tkinter import filedialog
import rasterio
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.models import Model

def preprocess_image(raw_img, preprocess_input):
    img = preprocess_input(np.moveaxis(raw_img, 0, -1))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def process_expanded_images(images, model):
    return [model.predict(np.expand_dims(preprocess_image(img, preprocess_input), axis=0))[0] for img in images]

def process_image(model, image_path):
    with rasterio.open(image_path) as src:
        raw_img = src.read()
        
    expanded_images = [np.copy(raw_img)]
    
    rotated = np.copy(raw_img)
    for i in range(3):
        for c in range(rotated.shape[0]):
            rotated[c, :, :] = cv2.rotate(rotated[c, :, :], cv2.ROTATE_90_CLOCKWISE)
        expanded_images.append(np.copy(rotated))

    predictions = process_expanded_images(expanded_images, model)
    masks = [np.argmax(pred, axis=-1) for pred in predictions]
    
    rotated_masks = []

    for i in range(4):
        rotate = np.copy(masks[i])
        for _ in range(i):
            rotate = cv2.rotate(rotate, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_masks.append(np.copy(rotate))

    average_mask = np.mean(rotated_masks, axis=0)
    final_mask = (average_mask > 0).astype(np.uint8)

    plt.figure(figsize=(12, 4))
    for i, mask in enumerate(rotated_masks, 0):
        plt.subplot(1, len(rotated_masks), i+1)
        plt.imshow(np.transpose(raw_img, (1, 2, 0))[:, :, :3])
        plt.imshow(rotated_masks[i], cmap='viridis', alpha=0.5, vmin=0, vmax=1)
        plt.title(f'Mask {i+1}')

    plt.show()

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(np.transpose(raw_img, (1, 2, 0))[:, :, :3], cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(final_mask, cmap='viridis', vmin=0, vmax=1)
    plt.title('Predicted Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(np.transpose(raw_img, (1, 2, 0))[:, :, :3], cv2.COLOR_BGR2RGB))
    plt.imshow(final_mask, cmap='viridis', alpha=0.3, vmin=0, vmax=1)
    plt.title('Overlay')

    plt.show()

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif;*.png")])
    if file_path:
        process_image(model, file_path)

BACKBONE = 'mobilenet'
IMG_HEIGHT = 256
IMG_WIDTH = 256
N_CLASSES = 2
preprocess_input = sm.get_preprocessing(BACKBONE)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    base_model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), classes=N_CLASSES, activation='softmax')
    base_model.compile('Adam', loss=sm.losses.cce_jaccard_loss, metrics=[sm.metrics.iou_score])
model = Model(inputs=base_model.input, outputs=base_model.output)

weights_path = 'checkpoints/checkpoint.h5'
model.load_weights(weights_path)

root = tk.Tk()
root.title("Image Segmentation App")
root.geometry("400x200")  # Увеличиваем размер окна

button = tk.Button(root, text="Select Image", command=browse_file)
button.pack(side="top", expand=True, pady=50)

root.mainloop()
