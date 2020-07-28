import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
from PIL import Image

image_size = 224

def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image, label

def process_image(numpy_image):
    
    tensor_image = tf.convert_to_tensor(numpy_image)
    
    tf.convert_to_tensor(tensor_image, dtype=None, dtype_hint=None, name=None)
    
    tf.image.resize(tensor_image, [224,224]).shape.as_list()
    
    tf.image.per_image_standardization(tensor_image)
    
    numpy_image = tensor_image.numpy()

    return numpy_image

def predict(image_path, model, topk):
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)

    topk = int(topk)
        
    loaded_model = tf.keras.models.load_model(model,custom_objects={'KerasLayer':hub.KerasLayer})
        
    image = Image.open(image_path)
    image = np.asarray(image)

    processed_test_image = process_image(image)
    processed_test_image, label = format_image(processed_test_image, 'label')
    processed_test_image = np.expand_dims(processed_test_image, axis=0)

    tensor_image = tf.convert_to_tensor(processed_test_image)

    probs = loaded_model.predict(tensor_image)
    probs = np.array(probs[0])

    idxs = np.argpartition(probs, -topk)[-topk:]
    probs = probs[np.argpartition(probs, -topk)[-topk:]]

    counter = 0
    for idx, val in enumerate(idxs):
        print(class_names.get(str(val+1)) + ' : ' + str(probs[counter]))
        counter += 1
        
predict(sys.argv[1], sys.argv[2], sys.argv[3])