import tensorflow as tf
import os
import numpy as np
import cv2
import pickle
from random import shuffle
from matplotlib import pyplot as plt

TRAIN_DIR = '../train'
TEST_DIR = '../test1'
CAT_LBL = 0
DOG_LBL = 1
LABEL_MAP = {
    'cat': CAT_LBL,
    'dog': DOG_LBL
}
DATA_SIZE = 18_000
IMG_SIZE = 110
SPLIT_RATIO = 0.8


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(96, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(96, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Model prepared...')
    return model


def label_img(name):
    word_label = name.split('.')[0]
    label = LABEL_MAP[word_label]
    label_arr = np.zeros(2)
    label_arr[label] = 1
    return label_arr


def prep_and_load_data():
    data = []
    image_paths = os.listdir(TRAIN_DIR)
    shuffle(image_paths)
    count = 0
    for img_path in image_paths:
        label = label_img(img_path)
        path = os.path.join(TRAIN_DIR, img_path)
        image = cv2.imread(path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype('float') / 255.0
        data.append([image, label])
        count += 1
        print(f'Processing image {count}/{DATA_SIZE}')
        if count == DATA_SIZE:
            break

    shuffle(data)
    print(f'Data preparation done. Total images: {len(data)}')
    return data


def plotter(history_file):
    with open(history_file, 'rb') as file:
        history = pickle.load(file)

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.show()

    # Loss plot
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()


def process_image(directory, img_path):
    path = os.path.join(directory, img_path)
    image = cv2.imread(path)
    image_copy = image.copy()
    
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_std = image.astype('float') / 255.0
    return image_copy, image_std

def image_inference(model):
    val_map = {1: 'Dog', 0: 'Cat'}
    image_paths = os.listdir(TEST_DIR)
    image_paths = image_paths[:200]

    results = []

    count = 0
    for img_path in image_paths:
        image, image_std = process_image(TEST_DIR, img_path)

        image_std = image_std.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        pred = model.predict([image_std])
        arg_max = np.argmax(pred, axis=1)
        max_val = np.max(pred, axis=1)
        result_text = f'{val_map[arg_max[0]]} - {max_val[0]*100:.2f}%'

        results.append((img_path, result_text))

        cv2.putText(image, result_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        result_img_path = os.path.join('predictions', f'{img_path}_prediction.jpg')
        os.makedirs('predictions', exist_ok=True)
        cv2.imwrite(result_img_path, image)

        count += 1
        print(f'Processed {count}/{len(image_paths)} images')

    return results


if __name__ == "__main__":
    data = np.array(prep_and_load_data())
    train_size = int(DATA_SIZE * SPLIT_RATIO)
    print(f'Data size: {len(data)}, Train size: {train_size}')

    train_data = data[:train_size]
    train_images = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    train_labels = np.array([i[1] for i in train_data])
    print('Training data fetched...')

    test_data = data[train_size:]
    test_images = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    test_labels = np.array([i[1] for i in test_data])
    print('Test data fetched...')

    model = get_model()
    print('Training started...')
    history = model.fit(train_images, train_labels, batch_size=50, epochs=15, verbose=1, validation_data=(test_images, test_labels))
    print('Training done.')

    model.save('final_model.h5')
    history_file = 'training_history.pickle'
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)

    plotter(history_file)

    results = image_inference(model)
    for img_path, result in results:
        print(f'Prediction for {img_path}: {result}')
