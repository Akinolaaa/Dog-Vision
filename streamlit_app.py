import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time

st.title("Find out the breed of your dog")

# Load model
model = tf.keras.models.load_model(
    "./20211124-00041637712298_full-image-set-mobilenetv2-Adam.h5",
    custom_objects={"KerasLayer": hub.KerasLayer},
)


def process_input(input_image):
    # Turn the jpeg image into numerical Tensor with 3 colour channels(rgb)
    # channels=1 output a grayscale image
    # channels=0; Use the number of channels in the JPEG-encoded image.
    image = tf.image.decode_jpeg(input_image.getvalue(), channels=3)
    # Convert the colour channel values from 0-255 to 0-1 values (scaling)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image to our desired value(224, 224)
    img_size = 224
    image = tf.image.resize(image, size=[img_size, img_size])
    image = np.array([image])
    image = tf.constant(image)
    data = tf.data.Dataset.from_tensor_slices(image)  # only file paths(no labels)
    data_batch = data.batch(32)

    return data_batch


def get_prediction(data):
    pred_array = model.predict(data)[0]
    np_breeds = pd.read_csv("./breeds.csv")["breeds"].to_numpy()
    predicted_breed = np_breeds[np.argmax(pred_array)]
    time.sleep(3)
    return predicted_breed


st.write("Upload an image of your dog: ")
dog_image_file = st.file_uploader("Choose a file", type=["png", "jpg"])
if dog_image_file is not None:
    data = process_input(dog_image_file)
    info = st.info("Calculating...", icon="ℹ️")
    predicted_breed = get_prediction(data)
    st.markdown(
        "[View more images](https://www.google.com/search?q="
        + predicted_breed
        + "&tbm=isch)"
    )
    info.info("This dog is most likely a " + predicted_breed, icon="ℹ️")
    # Display image
    st.image(dog_image_file.getvalue())  # has caption
