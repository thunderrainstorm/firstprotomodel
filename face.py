'''
file_pattern = 'data/images/*.jpg'

# Load the dataset of file paths
images = tf.data.Dataset.list_files(file_pattern)

# Convert to numpy iterator and get the first file path
try:
    image_path = next(iter(images))
    print("First image path:", image_path)
except StopIteration:
    print("No matching files found for pattern:", file_pattern)


def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

# Assume 'images' is your TensorFlow Dataset object loaded with image file paths
images = images.map(load_image)

# Convert to numpy iterator and get the first image
first_image = next(iter(images.as_numpy_iterator()))
print("Shape of the first image:", first_image.shape)
print(type(images))

image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()


folders = ['train', 'test', 'val']

# Iterate through each folder(this one till alb is to be kept in comments
for folder in folders:
    # Get the list of files in the 'data/<folder>/images' directory
    image_dir = os.path.join('data', folder, 'images')
    for file in os.listdir(image_dir):
        # Construct the new JSON file path
        filename = file.split('.')[0] + '.json'
        existing_filepath = os.path.join('data', 'labels', filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join('data', folder, 'labels', filename)
            # Replace the existing JSON file with the new file path
            os.replace(existing_filepath, new_filepath)
'''

##div
'''
import os
import json
import numpy as np
import cv2
import albumentations as alb
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the augmentor
augmentor = alb.Compose([
    alb.RandomCrop(width=450, height=450),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)
], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))


# Function to augment and save images
def augment_and_save_images(data_dir, output_dir, partitions=['train', 'test', 'val']):
    for partition in partitions:
        partition_dir = os.path.join(data_dir, partition)
        for image_name in os.listdir(os.path.join(partition_dir, 'images')):
            img_path = os.path.join(partition_dir, 'images', image_name)
            img = cv2.imread(img_path)

            coords = [0, 0, 0.00001, 0.00001]
            label_path = os.path.join(partition_dir, 'labels', f'{os.path.splitext(image_name)[0]}.json')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = json.load(f)
                coords = [
                    label['shapes'][0]['points'][0][0],
                    label['shapes'][0]['points'][0][1],
                    label['shapes'][0]['points'][1][0],
                    label['shapes'][0]['points'][1][1]
                ]
                coords = list(np.divide(coords, [640, 480, 640, 480]))

            try:
                for x in range(60):
                    augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                    aug_img_path = os.path.join(output_dir, partition, 'images',
                                                f'{os.path.splitext(image_name)[0]}.{x}.jpg')
                    cv2.imwrite(aug_img_path, augmented['image'])

                    annotation = {'image': image_name}
                    if os.path.exists(label_path):
                        if len(augmented['bboxes']) == 0:
                            annotation['bbox'] = [0, 0, 0, 0]
                            annotation['class'] = 0
                        else:
                            annotation['bbox'] = augmented['bboxes'][0]
                            annotation['class'] = 1
                    else:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0

                    aug_label_path = os.path.join(output_dir, partition, 'labels',
                                                  f'{os.path.splitext(image_name)[0]}.{x}.json')
                    with open(aug_label_path, 'w') as f:
                        json.dump(annotation, f)

            except Exception as e:
                print(e)


# Function to load images
def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


# Function to load labels
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)

    class_label = np.array([label['class']], dtype=np.uint8)
    bbox = np.array(label['bbox'], dtype=np.float32)
    return class_label, bbox


def load_labels_wrapper(label_path):
    class_label, bbox = tf.py_function(load_labels, [label_path], [tf.uint8, tf.float32])
    class_label.set_shape([1])
    bbox.set_shape([4])
    return class_label, bbox


# Function to prepare dataset
def prepare_dataset(image_pattern, label_pattern):
    images = tf.data.Dataset.list_files(image_pattern, shuffle=False)
    images = images.map(load_image)
    images = images.map(lambda x: tf.image.resize(x, (120, 120)))
    images = images.map(lambda x: x / 255)

    labels = tf.data.Dataset.list_files(label_pattern, shuffle=False)
    labels = labels.map(load_labels_wrapper)

    return images, labels


# Function to prepare train, test, and validation datasets
def prepare_train_test_val_datasets(data_dir):
    train_images, train_labels = prepare_dataset(f'{data_dir}/train/images/*.jpg', f'{data_dir}/train/labels/*.json')
    test_images, test_labels = prepare_dataset(f'{data_dir}/test/images/*.jpg', f'{data_dir}/test/labels/*.json')
    val_images, val_labels = prepare_dataset(f'{data_dir}/val/images/*.jpg', f'{data_dir}/val/labels/*.json')

    train_dataset = tf.data.Dataset.zip((train_images, train_labels)).shuffle(5000).batch(8).prefetch(4)
    test_dataset = tf.data.Dataset.zip((test_images, test_labels)).shuffle(1300).batch(8).prefetch(4)
    val_dataset = tf.data.Dataset.zip((val_images, val_labels)).shuffle(1000).batch(8).prefetch(4)

    return train_dataset, test_dataset, val_dataset


# Function to visualize samples
def visualize_samples(dataset, num_samples=4):
    data_samples = dataset.as_numpy_iterator()
    res = data_samples.next()

    fig, ax = plt.subplots(ncols=num_samples, figsize=(20, 20))
    for idx in range(num_samples):
        sample_image = res[0][idx]
        sample_coords = res[1][1][idx]

        if sample_image.max() <= 1.0:
            sample_image = (sample_image * 255).astype(np.uint8)
        else:
            sample_image = sample_image.astype(np.uint8)

        if sample_image.shape[-1] == 3:
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)

        sample_image_copy = sample_image.copy()
        cv2.rectangle(sample_image_copy,
                      tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)
        sample_image_copy = cv2.cvtColor(sample_image_copy, cv2.COLOR_BGR2RGB)
        ax[idx].imshow(sample_image_copy)

    plt.show()


if __name__ == "__main__":
    data_dir = 'data'
    output_dir = 'aug_data'

    # Create output directories if they do not exist
    for partition in ['train', 'test', 'val']:
        os.makedirs(os.path.join(output_dir, partition, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, partition, 'labels'), exist_ok=True)

    # Augment and save images
    augment_and_save_images(data_dir, output_dir)

    # Prepare datasets
    train_dataset, test_dataset, val_dataset = prepare_train_test_val_datasets(output_dir)

    # Visualize some samples
    visualize_samples(train_dataset)

'''
##Div

'''
import albumentations as alb
augmentor = alb.Compose([
    alb.RandomCrop(width=450, height=450),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)
], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))
img_path = os.path.join('data', 'train', 'images', 'f644af26-2c9b-11ef-89bd-6c2b59d4dbba.jpg')
img = cv2.imread(img_path)

# Define the label file path using the same variable names
label_path = os.path.join('data', 'train', 'labels', 'f644af26-2c9b-11ef-89bd-6c2b59d4dbba.json')
with open(label_path, 'r') as f:
    label = json.load(f)

# Accessing points from label data
points = label['shapes'][0]['points']
print(points)
coords = [0, 0, 0, 0]

# Assign values to 'coords' using the same variable names as provided
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]
print(coords)
coords = list(np.divide(coords, [640,480,640,480]))
print(coords)
augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
print(augmented['bboxes']) #cropped coordinated of xmin,ymin,xmax,ymax
print(augmented['bboxes'][0][2:]) #cropped coordinated of xmax,ymax
# Draw rectangle on 'augmented['image']' using the same variable names as provided
cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2], [450, 450]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [450, 450]).astype(int)),
              (0, 255, 0), 2)

# Display the image using matplotlib
plt.imshow(cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB))
plt.show()




#17/6/2024
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast


# Define the augmentation pipeline
def augmentor(image, bboxes, class_labels):
    augmentations = Compose([
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['class_labels']})

    augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
    return augmented


# Path to data directories
data_dir = 'data'
aug_data_dir = 'aug_data'
partitions = ['train', 'test', 'val']

# Ensure augmented data directories exist
for partition in partitions:
    os.makedirs(os.path.join(aug_data_dir, partition, 'images'), exist_ok=True)
    os.makedirs(os.path.join(aug_data_dir, partition, 'labels'), exist_ok=True)

# Separating all the images including augmented ones into the test, train, val of aug_data
for partition in partitions:
    images_path = os.path.join(data_dir, partition, 'images')
    labels_path = os.path.join(data_dir, partition, 'labels')

    for image_name in os.listdir(images_path):
        img_path = os.path.join(images_path, image_name)
        img = cv2.imread(img_path)

        coords = [0, 0, 0.00001, 0.00001]
        label_name = f'{os.path.splitext(image_name)[0]}.json'
        label_path = os.path.join(labels_path, label_name)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640, 480, 640, 480]))

        try:
            for x in range(60):  # Generate 60 augmented images for each original image
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                aug_image_name = f'{os.path.splitext(image_name)[0]}.{x}.jpg'
                aug_img_path = os.path.join(aug_data_dir, partition, 'images', aug_image_name)
                cv2.imwrite(aug_img_path, augmented['image'])

                annotation = {'image': image_name}

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0, 0, 0, 0]
                    annotation['class'] = 0

                aug_label_name = f'{os.path.splitext(image_name)[0]}.{x}.json'
                aug_label_path = os.path.join(aug_data_dir, partition, 'labels', aug_label_name)
                with open(aug_label_path, 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)

def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

# Paths to images
train_images = tf.data.Dataset.list_files('aug_data/train/images/*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
train_images = train_images.map(lambda x: x / 255)

test_images = tf.data.Dataset.list_files('aug_data/test/images/*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120, 120)))
test_images = test_images.map(lambda x: x / 255)

val_images = tf.data.Dataset.list_files('aug_data/val/images/*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120, 120)))
val_images = val_images.map(lambda x: x / 255)

# Show a sample image
print(train_images.as_numpy_iterator().next())

# Function to load labels
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)
    class_label = np.array([label['class']], dtype=np.uint8)
    bbox = np.array(label['bbox'], dtype=np.float32)
    return class_label, bbox

def load_labels_wrapper(label_path):
    class_label, bbox = tf.py_function(load_labels, [label_path], [tf.uint8, tf.float32])
    class_label.set_shape([1])
    bbox.set_shape([4])
    return class_label, bbox

# Paths to labels
train_labels = tf.data.Dataset.list_files('aug_data/train/labels/*.json', shuffle=False)
train_labels = train_labels.map(load_labels_wrapper)
test_labels = tf.data.Dataset.list_files('aug_data/test/labels/*.json', shuffle=False)
test_labels = test_labels.map(load_labels_wrapper)
val_labels = tf.data.Dataset.list_files('aug_data/val/labels/*.json', shuffle=False)
val_labels = val_labels.map(load_labels_wrapper)

# Show a sample label
print(train_labels.as_numpy_iterator().next())

# Dataset lengths
print(len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels))

# Create and prepare datasets
train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)

# Show shapes of the dataset items
print(train.as_numpy_iterator().next()[0].shape)
print(train.as_numpy_iterator().next()[1])

# Display sample images with bounding boxes
data_samples = train.as_numpy_iterator()
res = data_samples.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))

for idx in range(4):
    sample_image = res[0][idx]
    sample_coords = res[1][1][idx]

    # Ensure the image is in the correct range and type
    if sample_image.max() <= 1.0:
        sample_image = (sample_image * 255).astype(np.uint8)
    else:
        sample_image = sample_image.astype(np.uint8)

    # Convert image from RGB to BGR for OpenCV
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)

    # Make a writable copy of the image
    sample_image_copy = sample_image.copy()

    # Draw the rectangle on the image copy
    cv2.rectangle(sample_image_copy,
                  tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                  (255, 0, 0), 2)

    # Convert the image back to RGB for matplotlib
    sample_image_copy = cv2.cvtColor(sample_image_copy, cv2.COLOR_BGR2RGB)

    # Display the image with the rectangle
    ax[idx].imshow(sample_image_copy)

plt.show()

'''

###################################################################################################################
'''

import os
import json
import numpy as np
import cv2
import albumentations as alb
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

# Define the augmentor
augmentor = alb.Compose([
    alb.RandomCrop(width=450, height=450),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)
], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))


# Function to augment and save images
def augment_and_save_images(data_dir, output_dir, partitions=['train', 'test', 'val']):
    for partition in partitions:
        partition_dir = os.path.join(data_dir, partition)
        for image_name in os.listdir(os.path.join(partition_dir, 'images')):
            img_path = os.path.join(partition_dir, 'images', image_name)
            img = cv2.imread(img_path)

            coords = [0, 0, 0.00001, 0.00001]
            label_path = os.path.join(partition_dir, 'labels', f'{os.path.splitext(image_name)[0]}.json')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = json.load(f)
                coords = [
                    label['shapes'][0]['points'][0][0],
                    label['shapes'][0]['points'][0][1],
                    label['shapes'][0]['points'][1][0],
                    label['shapes'][0]['points'][1][1]
                ]
                coords = list(np.divide(coords, [640, 480, 640, 480]))

            try:
                for x in range(60):
                    augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                    aug_img_path = os.path.join(output_dir, partition, 'images',
                                                f'{os.path.splitext(image_name)[0]}.{x}.jpg')
                    cv2.imwrite(aug_img_path, augmented['image'])

                    annotation = {'image': image_name}
                    if os.path.exists(label_path):
                        if len(augmented['bboxes']) == 0:
                            annotation['bbox'] = [0, 0, 0, 0]
                            annotation['class'] = 0
                        else:
                            annotation['bbox'] = augmented['bboxes'][0]
                            annotation['class'] = 1
                    else:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0

                    aug_label_path = os.path.join(output_dir, partition, 'labels',
                                                  f'{os.path.splitext(image_name)[0]}.{x}.json')
                    with open(aug_label_path, 'w') as f:
                        json.dump(annotation, f)

            except Exception as e:
                print(e)


# Function to load images
def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


# Function to load labels
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)

    class_label = np.array([label['class']], dtype=np.uint8)
    bbox = np.array(label['bbox'], dtype=np.float32)
    return class_label, bbox


def load_labels_wrapper(label_path):
    class_label, bbox = tf.py_function(load_labels, [label_path], [tf.uint8, tf.float32])
    class_label.set_shape([1])
    bbox.set_shape([4])
    return class_label, bbox


# Function to prepare dataset
def prepare_dataset(image_pattern, label_pattern):
    images = tf.data.Dataset.list_files(image_pattern, shuffle=False)
    images = images.map(load_image)
    images = images.map(lambda x: tf.image.resize(x, (120, 120)))
    images = images.map(lambda x: x / 255)

    labels = tf.data.Dataset.list_files(label_pattern, shuffle=False)
    labels = labels.map(load_labels_wrapper)

    return images, labels


# Function to prepare train, test, and validation datasets
def prepare_train_test_val_datasets(data_dir):
    train_images, train_labels = prepare_dataset(f'{data_dir}/train/images/*.jpg', f'{data_dir}/train/labels/*.json')
    test_images, test_labels = prepare_dataset(f'{data_dir}/test/images/*.jpg', f'{data_dir}/test/labels/*.json')
    val_images, val_labels = prepare_dataset(f'{data_dir}/val/images/*.jpg', f'{data_dir}/val/labels/*.json')

    train_dataset = tf.data.Dataset.zip((train_images, train_labels)).shuffle(5000).batch(8).prefetch(4)
    test_dataset = tf.data.Dataset.zip((test_images, test_labels)).shuffle(1300).batch(8).prefetch(4)
    val_dataset = tf.data.Dataset.zip((val_images, val_labels)).shuffle(1000).batch(8).prefetch(4)

    return train_dataset, test_dataset, val_dataset


# Function to visualize samples
def visualize_samples(dataset, num_samples=4):
    data_samples = dataset.as_numpy_iterator()
    res = data_samples.next()

    fig, ax = plt.subplots(ncols=num_samples, figsize=(20, 20))
    for idx in range(num_samples):
        sample_image = res[0][idx]
        sample_coords = res[1][1][idx]

        if sample_image.max() <= 1.0:
            sample_image = (sample_image * 255).astype(np.uint8)
        else:
            sample_image = sample_image.astype(np.uint8)

        if sample_image.shape[-1] == 3:
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)

        sample_image_copy = sample_image.copy()
        cv2.rectangle(sample_image_copy,
                      tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)
        sample_image_copy = cv2.cvtColor(sample_image_copy, cv2.COLOR_BGR2RGB)
        ax[idx].imshow(sample_image_copy)

    plt.show()


# Build the model
def build_model():
    input_layer = Input(shape=(120, 120, 3))

    vgg = VGG16(include_top=False)(input_layer)
    # Classification Model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    # Regression model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker


# Custom localization loss function
def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(
        tf.square(y_true[:, :2] - yhat[:, :2]))  # Distance between actual and predicted coordinates
    h_true = y_true[:, 3] - y_true[:, 1]  # Actual height
    w_true = y_true[:, 2] - y_true[:, 0]  # Actual width

    h_pred = yhat[:, 3] - yhat[:, 1]  # Predicted height
    w_pred = yhat[:, 2] - yhat[:, 0]  # Predicted width

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))  # Size difference

    return delta_coord + delta_size  # Localization loss


# Custom FaceTracker model class
class FaceTracker(Model):
    def __init__(self, facetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = facetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch
        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            total_loss = batch_localizationloss + 0.5 * batch_classloss
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch
        classes, coords = self.model(X, training=False)
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)


if __name__ == "__main__":
    data_dir = 'data'
    output_dir = 'aug_data'

    # Create output directories if they do not exist
    for partition in ['train', 'test', 'val']:
        os.makedirs(os.path.join(output_dir, partition, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, partition, 'labels'), exist_ok=True)

    # Augment and save images
    augment_and_save_images(data_dir, output_dir)

    # Prepare datasets
    train_dataset, test_dataset, val_dataset = prepare_train_test_val_datasets(output_dir)

    # Visualize some samples
    visualize_samples(train_dataset)

    # Build and compile the model
    facetracker = build_model()
    facetracker.summary()

    initial_learning_rate = 0.0001
    batches_per_epoch = len(train_dataset)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=batches_per_epoch,
        decay_rate=0.75,
        staircase=True
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    classloss = tf.keras.losses.BinaryCrossentropy()
    regressloss = localization_loss

    model = FaceTracker(facetracker)
    model.compile(opt, classloss, regressloss)

    logdir = 'pycharm logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    hist = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[tensorboard_callback])

    # Plot training history
    fig, ax = plt.subplots(ncols=3, figsize=(20, 5))

    ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
    ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
    ax[0].set_title('Loss')
    ax[0].legend()

    ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
    ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
    ax[1].set_title('Classification Loss')
    ax[1].legend()

    ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
    ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
    ax[2].set_title('Regression Loss')
    ax[2].legend()

    plt.show()

    # Test data visualization
    test_data = test_dataset.as_numpy_iterator()
    test_sample = next(test_data)
    yhat = facetracker.predict(test_sample[0])

    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx in range(4):
        sample_image = test_sample[0][idx]
        sample_coords = yhat[1][idx]

        sample_image_copy = sample_image.copy()

        if yhat[0][idx] > 0.9:
            cv2.rectangle(sample_image_copy,
                          tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                          tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                          (255, 0, 0), 2)

        ax[idx].imshow(sample_image_copy)

    plt.show()
'''
import os
import json
import numpy as np
import cv2
import albumentations as alb
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

    # Save and load model
#facetracker.save('facetracker.keras')
facetracker = load_model('facetracker.keras')

# Real-time video processing
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    frame = frame[20:460, 20:460, :]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                      (255, 0, 0), 2)

        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                   [0, -30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                   [80, 0])),
                      (255, 0, 0), -1)

        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                                [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Face', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

####################################################################################################################

'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16


def build_model():
    input_layer = Input(shape=(120, 120, 3))

    vgg = VGG16(include_top=False)(input_layer)
    # Classification Model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    # Regression model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker


# Custom localization loss function
def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(
        tf.square(y_true[:, :2] - yhat[:, :2]))  # Distance between actual and predicted coordinates
    h_true = y_true[:, 3] - y_true[:, 1]  # Actual height
    w_true = y_true[:, 2] - y_true[:, 0]  # Actual width

    h_pred = yhat[:, 3] - yhat[:, 1]  # Predicted height
    w_pred = yhat[:, 2] - yhat[:, 0]  # Predicted width

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))  # Size difference

    return delta_coord + delta_size  # Localization loss


# Custom FaceTracker model class
class FaceTracker(Model):
    def __init__(self, facetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = facetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch
        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            total_loss = batch_localizationloss + 0.5 * batch_classloss
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch
        classes, coords = self.model(X, training=False)
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)



    # Build and compile the model
    facetracker = build_model()
    facetracker.summary()

    initial_learning_rate = 0.0001
    batches_per_epoch = len(train_dataset)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=batches_per_epoch,
        decay_rate=0.75,
        staircase=True
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    classloss = tf.keras.losses.BinaryCrossentropy()
    regressloss = localization_loss

    model = FaceTracker(facetracker)
    model.compile(opt, classloss, regressloss)

    logdir = 'pycharm logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

 #  hist = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[tensorboard_callback])

'''
'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

# Load the VGG16 model without the top classification layers
vgg = VGG16(include_top=False)

# Print the summary of the VGG16 model to understand its architecture
vgg.summary()


def build_model():  # Deep learning model
    input_layer = Input(shape=(120, 120, 3))

    # Load the VGG16 model without the top classification layers
    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    # Regression Model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    # Combine both models into one model
    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker


# Create the model
facetracker = build_model()

# Print the summary of the model to understand its architecture
facetracker.summary()
X, y = train.as_numpy_iterator().next()

# Check the shape of the input data
print(X.shape)

# Make predictions using the facetracker model
classes, coords = facetracker.predict(X)

# Display the predictions
print('Classes:', classes)
print('Coordinates:', coords)
batches_per_epoch = len(train)
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=batches_per_epoch,
    decay_rate=0.75,
    staircase=True
)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))

    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]

    h_pred = yhat[:, 3] - yhat[:, 1]
    w_pred = yhat[:, 2] - yhat[:, 0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    return delta_coord + delta_size


classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss


# Example usage of loss functions
# Assuming `y[0]` is for classes and `y[1]` is for coordinates
# localization_loss(y[1], coords)
# classloss(y[0], classes)
# regressloss(y[1], coords)

class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

            total_loss = batch_localizationloss + 0.5 * batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)


# Example instantiation and compilation of FaceTracker model
# Assuming `facetracker` is already defined as the VGG16 model
model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)

logdir = 'pycharm logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback]) # we'll be able to get our training history here
'''
