

from __future__ import print_function

import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Rescaling,
    GlobalAveragePooling2D,
)
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight



batch_size = 12
num_classes = 3
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True #make fit false if you do not want to train the network again

train_dir = r"C:\Users\B00144121\Downloads\chest_xray\chest_xray\train"
test_dir = r"C:\Users\B00144121\Downloads\chest_xray\chest_xray\test"
def count_images(path):
    counts = {}
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            counts[class_name] = len(
                [f for f in os.listdir(class_path)
                 if os.path.isfile(os.path.join(class_path, f))]
            )
    return counts

print("Train distribution:", count_images(train_dir))
print("Test distribution:", count_images(test_dir))


with tf.device('/gpu:0'):
    
    #create training,validation and test datatsets
    train_ds,val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)
    
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)
   


    class_names = train_ds.class_names
    print('Class Names: ',class_names)
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(min(6, len(images))):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.tight_layout()       
    plt.show()

    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
    ])
    #create model
    model = tf.keras.models.Sequential([
        Rescaling(1.0/255),
        Conv2D(16, (3,3), activation = 'relu', input_shape = (img_height,img_width, img_channels)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation = 'relu'),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation = 'relu'),
        MaxPooling2D(2,2),
         # flatten multidimensional outputs into single dimension for input to dense fully connected layers
         GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                    metrics=['accuracy',
                   tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                    ]
                    
    )

    model.summary()

    #
    # Calculate class weights 
    train_labels = []
    for _, labels in train_ds.unbatch():
        train_labels.append(labels.numpy())

    class_weights = compute_class_weight(
        class_weight='balanced',    
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weights_dict)

    save_callback = tf.keras.callbacks.ModelCheckpoint(
        "pneumonia.keras", 
          save_best_only=True,
          monitor='val_loss', 
          mode='min'
    )
  
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
    )
    

    if fit:
        start_time = time.time()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            callbacks=[save_callback, earlystop_callback],
            epochs=epochs,
            class_weight=class_weights_dict
        )
        end_time = time.time()
        print("Training time:", round(end_time - start_time, 2), "seconds")
    else:
        model = tf.keras.models.load_model("pneumonia.keras")

    #if shuffle=True when creating the dataset, samples will be chosen randomly   
    score = model.evaluate(test_ds, batch_size=batch_size)
    print('Test accuracy:', score[1])

    
    if fit:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        
    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(),0))#perform a prediction on this image
            plt.title('Actual:' + class_names[labels[i].numpy()]+ '\nPredicted:{} {:.2f}%'.format(class_names[np.argmax(prediction)], 100 * np.max(prediction)))
            plt.axis("off")
    plt.show()
