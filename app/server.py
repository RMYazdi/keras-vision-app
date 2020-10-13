

import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization
from keras.utils import np_utils
from keras.utils import model_to_dot
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop
from keras.models import Sequential, model_from_json
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers.merge import concatenate
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
import os
from glob import glob
import random
import cv2
import pandas as pd
import numpy as np
import itertools
import sklearn
import scipy
import skimage
from skimage.transform import resize
from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from keras.applications.inception_v3 import decode_predictions









from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn, aiohttp, asyncio
import sys, numpy as np

path = Path(__file__).parent
model_file_url = 'https://drive.google.com/uc?export=download&id=1-4D7hdrEEiKsnj1gjr99XVhqgfwBm4y8'
model_file_name = 'model'

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

MODEL_PATH = path/'models'/f'{model_file_name}.h5'
IMG_FILE_SRC = '/tmp/saved_image.png'















# Defining Hyperparameters and model structure
image_size=224
target_dims = (image_size,image_size,3) # add channel for RGB
number_classes = 2
number_epoch=150
batch_size=32
learningRate=0.0001
patience=100
# optimizer = keras.optimizers.Adam(lr=0.001)
optimizer = keras.optimizers.RMSprop(lr=learningRate)









#Generare Model

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

def Model_build():

  base_model = InceptionV3(include_top=False, weights='imagenet')
  # Training only top layers i.e. the layers which we have added in the end
  for layer in base_model.layers:
      layer.trainable = True
  
  # Taking the output of the last convolution block in InceptionV3
  x = base_model.output
  
  # Adding a Global Average Pooling layer
  x = GlobalAveragePooling2D()(x)



  # Adding a fully connected layer having 2 neurons which will
  # give the probability of image having either dog or cat
  predictions = Dense(2, activation='softmax')(x)
  
  # Model to be trained
  model = Model(inputs=base_model.input, outputs=predictions)
    
    # Training only top layers i.e. the layers which we have added in the end
  return model


  









































async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_model():
 
  
    
    #UNCOMMENT HERE FOR CUSTOM TRAINED MODEL
    await download_file(model_file_url, MODEL_PATH)
    model=Model_build()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    model.load_weights(MODEL_PATH)

    
#     model = load_model(MODEL_PATH) # Load your Custom trained model
    # model._make_predict_function()
#     model = ResNet50(weights='imagenet') # COMMENT, IF you have Custom trained model
    return model

# Asynchronous Steps
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["file"].read())
    with open(IMG_FILE_SRC, 'wb') as f: f.write(img_bytes)
    return model_predict(IMG_FILE_SRC, model)





def model_predict(img_path, model):
    result = [];     
    
    img = cv2.imread(img_path)
    img = skimage.transform.resize(img, (224, 224))
    img_file=img/255.0
    x = np.asarray(img_file)
    x=x.reshape((1,224,224,3))
    y_pred = model.predict(x)
    
    
    Label_dict={0:"Patient",1:"Normal"}
    label=Label_dict[np.argmax(y_pred, axis=1)[0]]
    accuracy=max(y_pred[0])
    
    result.append((label,accuracy))
    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    result_html = str(result_html1.open().read() +str(result) + result_html2.open().read())
    return HTMLResponse(result_html)







# def model_predict(img_path, model):
#     result = []; img = image.load_img(img_path, target_size=(224, 224))
#     x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
#     predictions = decode_predictions(model.predict(x), top=3)[0] # Get Top-3 Accuracy
#     for p in predictions: _,label,accuracy = p; result.append((label,accuracy))
#     result_html1 = path/'static'/'result1.html'
#     result_html2 = path/'static'/'result2.html'
#     result_html = str(result_html1.open().read() +str(result) + result_html2.open().read())
#     return HTMLResponse(result_html)

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
