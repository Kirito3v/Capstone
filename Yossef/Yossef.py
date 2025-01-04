

import os
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.get_logger().setLevel('ERROR')

import numpy as np
import faiss 
import sys
from tf_keras.applications import MobileNetV2
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D
from tf_keras.models import Model,load_model
from tf_keras.optimizers import Adam


folder = "Dataset Step2 - 1Image\keyboard"

train_dir = folder + '/train' 

input_shape = (512, 512, 3) 

image_path = "tests/tg.jpg"  #test image

batch_size = 32
epochs = 50

datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=180,      
    width_shift_range=0.2,   
    height_shift_range=0.2,  
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True,    
    validation_split=0.2     
)




train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
)

iter = 100
    
num_samples =  train_generator.samples * iter


if not os.path.exists('features.npy') or not os.path.exists('features_model.keras') or not os.path.exists('features_labels.npy'):

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    base_model.trainable = False


    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = Dense(128, activation='relu')(x)  
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    output = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)


    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    print(train_generator.n)
    print(train_generator.n//batch_size)

    model.fit(train_generator, epochs=epochs)


    test_loss, test_accuracy = model.evaluate(train_generator)

    print("Loss:", test_loss)
    print("Accuracy:", test_accuracy)
    
    
    all_images = []
    all_labels = []


    for images, labels in train_generator:
        all_images.extend(images)
        all_labels.extend(labels)    
        if len(all_images) >= num_samples :
            break
            
    all_images = np.array(all_images)


    features_extractor = Model(inputs=base_model.input, outputs=model
                .layers[-1].output)
    features = features_extractor.predict(all_images)

    features_extractor.save('features_model.keras')

    np.save('features.npy', features)
    np.save('features_labels.npy',all_labels)
    np.save('allimages.npy',all_images)
    

all_labels =  np.load('features_labels.npy')
features_extractor = load_model('features_model.keras')
features = np.load('features.npy')
all_images = np.load('allimages.npy')

index = faiss.IndexFlatL2(features.shape[1])
index.add(features)



img = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape)
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0 
img_array = np.expand_dims(img_array, axis=0)
Img_features = features_extractor.predict(img_array).flatten()


D, I = index.search(np.array([Img_features]), k=num_samples)


import matplotlib.pylab as plt

k = len(train_generator.class_indices)

fig, axes = plt.subplots(1, k, figsize=(10, 5))

fig.suptitle(f"Search results for '{os.path.basename(image_path)}'", fontsize=16)

class_indices = train_generator.class_indices  
classes = {v: k for k, v in class_indices.items()}  


i =0

added_names = []

for idx in I[0]:
    
    predicted_label = np.argmax(all_labels[idx])
    
    prediction_name = classes.get(predicted_label, "Unknown")
    
    if(not added_names.__contains__(prediction_name)):
        added_names.append(prediction_name)
        
        img = all_images[idx]
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{i+1}-{prediction_name} (Dist: {D[0][i]:.3f})")
        i+=1
    if(i>k-1):
        break
    

plt.tight_layout()

plt.show()

