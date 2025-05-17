# Celebrity Face Recognition using Deep Transfer Learning (AlexNet + MATLAB)

Project Overview
Celebrity Face Recognition is a deep learning-based computer vision project developed using MATLAB and powered by AlexNet via Transfer Learning. The system is designed to accurately classify and recognize celebrity faces from a custom dataset using a pre-trained convolutional neural network (CNN) architecture, adapted for the specific domain of facial identity classification.

The core objective is to demonstrate how transfer learning on AlexNet can be effectively applied to real-world image recognition tasks, especially when labeled data is limited.

Key Features
Deep Transfer Learning
Utilizes a pre-trained AlexNet model, fine-tuned with customized classification layers to learn celebrity face embeddings.

Dataset Management & Preprocessing
Organizes image data using MATLAB’s imageDatastore, with subfolder-based label inference. Implements data augmentation techniques such as random rotation, scaling, and translation to improve generalization.

Data Splitting and Augmentation
Splits the dataset into 70% training, 15% validation, and 15% test sets. Applies augmentedImageDatastore to standardize input and introduce variability.

Performance Evaluation Metrics
Evaluates model accuracy, precision, recall, and confusion matrices on both validation and test sets. Achieves high classification accuracy (>89%) across diverse identities.

Inference on Unseen Data
Enables prediction on new facial images with a confidence score, showcasing the model’s ability to generalize to real-world test inputs.

Technical Stack
Framework: MATLAB Deep Learning Toolbox

Model: AlexNet (modified classifier head)

Libraries: imageDatastore, augmentedImageDatastore, confusionchart, trainNetwork, classify

Metrics: Accuracy, Confusion Matrix, Precision, Recall

Data Augmentation: Random rotation, X/Y translation, scaling, RGB conversion from grayscale

Results
Validation Accuracy: 88.4%

Test Accuracy: 89.7%

Precision & Recall: Evaluated per class and overall

Live Prediction: Achieved 100% confidence on sample celebrity image ("Maria Sharapova")

Use Cases
Automated celebrity recognition systems

Face-based access control and identification

Demonstration tool for academic deep learning workflows

Foundation for further extension into real-time applications or mobile deployment

Future Enhancements
Integrate with real-time webcam feeds for live recognition

Expand dataset with more celebrity classes and ethnic/gender diversity

Explore advanced architectures like ResNet, EfficientNet, or Vision Transformers

Deploy model as a RESTful API using MATLAB Compiler SDK or Python/Node backend


