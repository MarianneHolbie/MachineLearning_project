# Transfer Learning with Convolutional Neural Networks (CNNs)

This repository contains a Python script for training a Convolutional Neural Network (CNN) using transfer learning to classify the CIFAR-10 dataset with Keras. 
Additionally, it includes a preprocessing function to prepare the data for training.

## Description
[0-transfer.py](./0-transfer.py)
This Python script trains a CNN using transfer learning with the Keras framework. 

* The Keras Applications module is used to load a pre-trained model ([InceptionResNetV2](https://keras.io/api/applications/inceptionresnetv2/)) that has been trained on the ImageNet dataset.
* The trained model is saved as `cifar10.h5`.
* The saved model is compiled.
* Achieves a validation accuracy of **94%**.
* Lambda layer is used to scale up the input data to the correct size.
* Most of the application layers are freeze to save computational resources.
* A function preprocess_data(X, Y) is provided to preprocess the data before training.
* Preprocessing Function (0-transfer.py)
The preprocess_data(X, Y) function preprocesses the CIFAR-10 dataset before training the model. It takes numpy arrays X and Y containing the dataset images and labels respectively, and returns preprocessed versions X_p and Y_p.


[0-main.py](./0-main.py)
The `0-main.py` file serves a crucial purpose in the project by providing a script 
to evaluate the performance of the pre-trained transfer learning model on the CIFAR-10 
dataset. This script imports the necessary functions from the `0-transfer.py` module, 
particularly the preprocess_data function, which preprocesses the CIFAR-10 data for 
input into the model. It then loads the CIFAR-10 dataset, preprocesses it using the 
imported function, loads the pre-trained model saved as `cifar10.h5`, and finally 
evaluates the model's performance on the preprocessed data. This evaluation step 
provides insights into how well the pre-trained model generalizes to the CIFAR-10 
dataset, which is essential for assessing its suitability for the task at hand. 
By running this script, users can quickly assess the effectiveness of the transfer 
learning approach in solving their image classification problem on CIFAR-10.

[cifar10.h5](./cifar10.h5)
The cifar10.h5 file provided in this repository contains a pre-trained model 
that has been customized and fine-tuned specifically for the CIFAR-10 dataset. 
Based on the InceptionResNetV2 architecture, this pre-trained model has undergone
enrichment and modifications tailored to CIFAR-10's characteristics. This adaptation
process involves refining the model's parameters, hyperparameters, and potentially
its architecture to better align with the features present in CIFAR-10 images. By 
customizing the pre-trained model for CIFAR-10, it ensures improved performance and 
effectiveness in classification tasks on this particular dataset, making it a valuable
asset for image recognition and related projects.

### Requirements
Python==3.9.13
tensorflow==2.6.0

### Blog Post

In this blog post, I delve into the experimental process of completing the task 
outlined above, employing transfer learning techniques to customize a pre-trained 
model for the CIFAR-10 dataset. 

The **abstract** succinctly summarizes the essence of the experimentation, 
encapsulating the objectives and outcomes in a concise manner.

The **introduction** section sets the stage by outlining the problem statement, 
elucidating the significance of adapting pre-trained models for specific datasets, 
and delineating the importance of transfer learning in the realm of deep learning. 

Moving on to the **materials and methods** section, I detail the experimental setup, 
including data preprocessing steps, model selection, and fine-tuning procedures. 

**Results** are presented in the subsequent section, highlighting the performance 
metrics and validation results of the customized model on the CIFAR-10 dataset. 

The **discussion** section interprets the findings, elucidating the implications of 
the experiment results and offering insights into the efficacy of transfer learning
in enhancing model performance for CIFAR-10 classification tasks. 


Throughout this [blog post](https://medium.com/@marianne.arrue/embarking-on-the-transfer-learning-journey-a-keras-adventure-with-cifar-10-332be2656ac3), I aim to provide a comprehensive overview of the 
transfer learning approach employed, offering insights, examples, and visual aids to
enhance comprehension and facilitate knowledge dissemination within the deep learning
community.

### Jupyter NoteBook
[Fine_tunning.ipynb](./Fine_tunning.ipynb) : JupyterNotebook that contains the code for fine tunning the model.
[Process_TransferLearning.ipynb](./Process_TransferLearning.ipynb) : JupyterNotebook that contains the code for the process of transfer learning.