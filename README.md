# Image-Counterfeit-Detection
Image counterfeit detection is the process of detecting and identifying the manipulation of  digital images. It is an important field of study for digital forensics, as it is used to detect and  analyze the tampering of digital images to identify the perpetrators of malicious activities. Image counterfeit can be accomplished by various methods, including copy-move, splicing, and retouching. Detection techniques for these methods include visual inspection, statistical analysis, and digital watermarking.

This project combines different deep learning techniques and image processing techniques to detect image tampering "Copy Move and Splicing" forgery in different image formats (either lossy or lossless formats). I implement two different techniques to detect tampering. 

I built my own model with ELA preprocessing and used fine tuning with two different pre-trained Models (VGG19, VGG16) which are trained using Google Colab, Image Counterfeitdetection application gives the user the ability to test images with the application-trained models or train the application model with new dataset and test images with this newly trained model. 

# Overview
The image counterfeit detection project is aimed at training an algorithm that can detect the presence of counterfeit images by analyzing their visual features. The project involves several stages, including data collection, preprocessing, feature extraction, model training, and evaluation.

In the first stage, a large dataset of authentic and counterfeit images is collected. The dataset should include a wide range of image types, such as photographs, logos, banknotes, and identity documents, and cover various levels of image quality and manipulations. The dataset is then preprocessed to remove any artifacts and standardize the image format, size, and orientation.

Next, image features such as color, texture, and shape are extracted from the preprocessed images. These features are then used to train a machine learning model or a neural network that can distinguish between authentic and counterfeit images. The model is trained on a subset of the dataset using a supervised learning approach, where the authentic and counterfeit labels are provided.

The trained model is then evaluated on a separate test dataset to assess its performance in detecting counterfeit images. The model is fine-tuned and optimized based on the evaluation results until it achieves high accuracy and robustness.

Finally, the developed image counterfeit detection system is deployed and integrated into asoftware tool that can be used by various stakeholders, such as law enforcement agencies, e-commerce platforms, and financial institutions. The system can also be updated and improved over time based on the feedback and performance data collected.

# Modular Description:

Model Training:

I.ELA:

The first step in the function is to take an image path and quality parameter as inputs. The function first resaves the input image with the specified quality and then calculates the difference between the original image and the resaved image. The resulting image is then scaled and returned as the ELA image. Next, the function reads the CSV file using Pandas and prepares the input and output data for the CNN. For each image in the CSV file, the function applies a function that is created with a quality parameter of 90 and resizes the resulting image to 128x128 pixels. The image is then flattened and normalized by dividing by 255. The label is also extracted and converted to one-hot encoding.The CNN model architecture consists of two convolutional layers, each followed by a max pooling layer and a dropout layer to prevent overfitting. The output of the convolutional layers is flattened and fed into two fully connected layers with dropout. The output layer has two units and uses the SoftMax activation function to output the probabilities for the two classes. The CNN is compiled using the categorical cross-entropy loss function and the specified optimizer. The model is then trained on the prepared data for the specified number of epochs and batch size. During training, the accuracy and loss of both the training and validation data are logged and plotted using Matplotlib. Finally, the function evaluates the trained model on the validation data and plots the confusion matrix using Matplotlib. The function also saves the trained model and the training history plot in separate files.

II.VGG16:

This code defines a function that takes three arguments: csv_file which is the path to a CSV file containing image data, lr which is the learning rate for the optimizer, and ep which is the number of epochs to train the model for. Inside the function, an image reading function called Read_image is defined using the PIL library. The CSV file is read using pandas, and the image data and labels are extracted and stored in X and Y variables respectively. The image data is preprocessed by resizing to 100x100 and flattening the pixel values before normalizing them to a range of 0 to 1. Next, the data is split into training and validation sets. Then, the VGG16 model is loaded from the Keras library with the pre-trained weights from the ImageNet dataset, but with the fully connected layer removed. Then, the layers of the VGG16 model are set as non-trainable except for the last 5 layers which will be fine-tuned during training. A new model is then created using the VGG16 as the base layer, followed by a Flatten layer and two Dense layers with ReLU and SoftMax activations respectively. The model is 
compiled with a binary cross-entropy loss function and an optimizer of RMSprop with the learning rate specified in the input argument.The model is trained, and the loss and accuracy values are plotted using matplotlib. Finally, the plot and the trained model are saved. The function returns the file path of the generated plot and the saved model, which can be used for further evaluation or testing.

III.VGG19:

The function reads the dataset from a CSV file and loads the images from the file path. It splits the dataset into training and validation sets and preprocesses the data by resizing the images to 100x100 and scaling the pixel values to the range [0,1]. The VGG19 model is loaded using the keras.applications.VGG19 function with the pre-trained weights from ImageNet. The model is then added to a new sequential model along with several fully connected layers, including a dropout layer to prevent overfitting. The last layer of the model has two nodes with a SoftMax activation function, which makes it suitable for binary classification tasks. It adds additional layers to the VGG19 model to adapt it to the new classification task. It compiles the model using the Adagrad optimizer and a mean squared error loss function and sets up a learning rate schedule using the step decay method. It trains the model using the fit method and saves the model and a plot of the loss and accuracy curves to files. The function returns the file paths of the saved plot and model.

Image input and Testing:

An image is given as input and it displays the information of the image then the model is tested with the given input image with the trained model in the previous step or we can test the image with the pre-trained model. The test_image_with_ela method applies an Error Level Analysis (ELA) algorithm to the input image and then feeds it into a pre-trained model to predict whether the image is forged or not. The test_image_with_vgg16 and test_image_with_vgg19 methods apply the VGG16 and VGG19 pre-trained models, respectively, to the input image to predict the same.

Image Pre-processing:

Image pre-processing is the third phase. Some pre-processing is performed on the picture under deliberation like image filtering, image enrichment, trimming, change in DCT coefficients, and RGB to grayscale transformation before handling the image to feature extraction procedures. Resizing the images to a standardized size can help to improve the efficiency of the algorithm used for counterfeit detection. Image compression is used to reduce the size of the images, making them easier to store and transmit. Enhancing the images can help to improve the clarity of the image and make it easier to detect counterfeits. Image segmentation can be used to separate the different parts of the image and focus on the regions that are important for counterfeit detection.

Feature Extraction: 

Feature extraction is an important step in image counterfeit detection. It involves identifying and extracting specific characteristics or features from the input image that can be used to distinguish between authentic and counterfeit images. The selection of features for every class separates the image set from different classes however in the meantime stays constant intended for a specific class chosen. Deep learning techniques such as Convolutional Neural Networks (CNNs) are used to extract features from images. CNNs have been shown to be highly effective in image classification tasks and can automatically learn features from images without the need for manual feature extraction. It involves re-saving the image with a high compression rate and comparing the original image with the compressed image to highlight areas that have been modified. These areas are then extracted as features for counterfeit detection. The attractive element of the International Journal of Computer chosen set of features is to have a tiny measurement so that computational complexity can be diminished and have a wide distinction from other classes. 

Classification: 

The only reason behind classification is to determine if the image is original or not. Neural systems is the classifier used for this purpose. Once you have extracted the features from your images, you can use various classification algorithms to train a model to detect counterfeit images, but I have used CNN classification. CNNs are a deep learning architecture that has achieved state-of-the-art performance in many image classification tasks, including counterfeit detection. They can automatically learn features from the images and are particularly effective when you have large amounts of data to train your model.

Postprocessing: 

Post-processing is the final step in image counterfeit detection, where the output of the classification model is analysed and adjusted for better accuracy. Some forgeries will possibly require post processing that include manipulations like localization of copy locales and after that the output is displayed. The result is a customized plot widget that creates a pie chart to visualize the prediction results of the image whether it is forged or not.
