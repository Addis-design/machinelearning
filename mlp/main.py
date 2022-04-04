from pyexpat import model
from re import A
from numpy import float32
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import math
from keras import layers
from tensorflow import keras
(ds_train,ds_test),ds_info=tfds.load(
    "mnist",
    split=["train","test"],
    shuffle_files=False,
    as_supervised=True,
    with_info=True,

)
@tf.function
def normalize_img(image,label):
    return tf.cast(image,tf.float32)/255.0,label

@tf.function
def rotate(img,max_degrees=25):
    math.degrees=tf.random.uniform([],-max_degrees,max_degrees,dtype=float32)
    img=tfa.image.rotate(img,math.degrees*math.pi/180,interpolation="BILINEAR")
    return img

@tf.function
def augment(image,label):
    image=tf.image.resize(image,size=[28,28])
    image=tf.image.random_contrast(image,lower=0.5,upper=1.5)
    image=tf.image.random_brightness(image,max_delta=0.2)
    image=rotate(image)
    #do argumentation
    return image,label
AUTOTUNE=tf.data.experimental.AUTOTUNE
BATCH_SIZE=32
ds_train=ds_train.cache()
ds_train=ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train=ds_train.map(normalize_img,num_parallel_calls=AUTOTUNE)
ds_train=ds_train.map(augment,num_parallel_calls=AUTOTUNE)
ds_train=ds_train.batch(BATCH_SIZE)
ds_train=ds_train.prefetch(AUTOTUNE)

ds_test=ds_test.map(normalize_img,num_parallel_calls=AUTOTUNE)
ds_test=ds_test.batch(BATCH_SIZE)
ds_test=ds_test.prefetch(AUTOTUNE)

def my_model():
    inputs=keras.Input(shape=(28,28,1))
    x=layers.Conv2D(32,3)(inputs)
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(64,3)(inputs)
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(128,3)(inputs)
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.Flatten()(x)
    x=layers.Dense(64,activation="relu")(x)
    outputs=layers.Dense(10,activation="softmax")(x)
    return keras.model(inputs=inputs,outputs=outputs)

model=my_model()
model.compile(
    loss=keras.losses.SparseCate
)
