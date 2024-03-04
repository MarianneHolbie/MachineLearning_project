# Neural Style Transfer

## Introduction
Neural Style Transfer is an optimization technique used to take three images, a content image, a style reference image (such as an artwork by a famous painter), and the input image you want to style -- and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.

## How it works
The principle of Neural Style Transfer (NST) is to define two distance functions, one that describes how different the content of two images are, Lcontent, and one that describes the difference between the two images in terms of their style, Lstyle. Then, given three images, a desired style image, a desired content image, and the input image (initialized with the content image), we try to transform the input image to minimize the content distance with the content image and its style distance with the style image.

## Implementation
The implementation of Neural Style Transfer is done using TensorFlow and Keras. The model used is VGG19, which is a pre-trained image classification model. The model is used to define the content and style representations of our images. These intermediate layers are necessary to define the representation of content and style from our images. For an input image, we try to match the corresponding style and content target representations at these intermediate layers.

## Tasks
| Task                                    | Description                                             |
|-----------------------------------------|---------------------------------------------------------|
| [Initialize](./0-neural_style.py)       | Initialize class NST and static method to rescale image |
| [Load Model](./1-neural_style.py)       | Load the model for neural style transfer |
| [Gram matrix](./2-neural_style.py)      | Calculate the gram matrix of a given matrix |
| [Extract Features](./3-neural_style.py) | Extract the features used to calculate neural style |
| [Layer Style Cost](./4-neural_style.py) | Calculate the style cost for a single layer |
| [Style Cost](./5-neural_style.py)       | Calculate the style cost |
| [Content Cost](./6-neural_style.py)     | Calculate the content cost |
| [Total Cost](./7-neural_style.py)       | Calculate the total cost |
| [Compute Gradients](./8-neural_style.py)| Compute the gradients for the neural style transfer |
| [Generate Image](./9-neural_style.py)   | Generate the neural style transfer image |
|[Variational Cost](./10-neural_style.py) | Calculate the total cost for the variational loss |
