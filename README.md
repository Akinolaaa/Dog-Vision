# Dog Breed Classification Model

This project consists of an image classification model capable of classifying a dog into one of 120 distinct breeds. The model is built using TensorFlow, a popular deep-learning framework.

## Model Architecture

The model architecture utilizes a pre-trained Keras layer from [tfhub.dev](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4) as the input layer. This layer leverages MobileNet V2, a lightweight convolutional neural network, to extract meaningful features from input images. The output layer is a Keras layer with softmax activation, enabling the model to assign probabilities to each of the 120 dog breed classes.

## Dataset

The dataset used for training and evaluation is obtained from [Kaggle's Dog Breed Identification competition](https://www.kaggle.com/c/dog-breed-identification/data). It consists of a large collection of dog images labeled with their respective breeds. The dataset provides an excellent resource for training and testing the model's performance.

## Model Training

During training, the model employs the Adam optimizer, a popular optimization algorithm, to minimize the categorical cross-entropy loss. With the given dataset and training configuration, the model achieves a loss of 0.0126 and an accuracy of 0.9984 after 100 epochs.

## Demo User Interface

To showcase the model's functionality, a demo user interface has been developed. You can access the UI using the following link: [Dog Vision Demo](https://ak-dog-vision.streamlit.app/). In this UI, users can upload dog images and observe the model's predictions in real time.

We hope you enjoy using the model and find it helpful in classifying dog breeds accurately!
