# Plant Disease Detection Web Application

This project is a web application that uses machine learning to detect diseases in plant leaves. Users can upload an image of a plant leaf, and the application will predict whether the plant is healthy or diseased.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)


## Features

- User-friendly web interface for uploading plant leaf images
- Real-time image preview before submission
- Machine learning model for disease detection
- Display of prediction results with the uploaded image

## Project Structure
This project structure is organized as follows:
This project structure is organized as follows:

- `app.py`: The main Flask application file.
- `model.h5`: The trained machine learning model for plant disease detection.
- `templates/`: Directory containing HTML templates.
  - `index.html`: The home page template for uploading images.
  - `result.html`: The result page template for displaying predictions.
- `static/`: Directory for static files.
  - `styles.css`: CSS file for styling the web pages.
  - `uploads/`: Directory to store uploaded images.
- `utils.py`: Utility functions for image processing and prediction.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file, containing project documentation.
- `Google_Colab_files/`: Directory containing files used for model training and development.
  - `Plant_Disease_Detection.ipynb`: Jupyter notebook with the model training process.
  - `train.csv`: Training data for the model.
  - `test.csv`: Test data for model evaluation.
  - `sample_submission.csv`: Sample submission file for the project.

## Model Training

The machine learning model used in this project was trained using Google Colab. You can find the notebook and associated files in the `Google_Colab_files` directory:

- `Plant_Disease_Detection.ipynb`: This Jupyter notebook contains the entire process of data preprocessing, model architecture design (Xception and DenseNet ensemble), training process, model evaluation, and saving the trained model.
- `train.csv`: The training dataset used for model development.
- `test.csv`: The test dataset used for model evaluation.
- `sample_submission.csv`: A sample submission file demonstrating the expected format of predictions.

To view and run the notebook:
1. Upload the `Plant_Disease_Detection.ipynb` file to Google Colab.
2. Upload the CSV files to the drive folder where collab notebook is present.
3. Follow the instructions in the notebook to train and evaluate the model.

Note: Due to the size of the datasets and the computational requirements, it's recommended to run this notebook in Google Colab with GPU acceleration enabled. 

## Installation
To get a local copy of the project up and running, follow these simple steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/chetankumar9903/plant-disease-detection.git 
   cd plant-disease-detection

2. Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

1. Run the Flask application:

2. Open a web browser and navigate to `http://localhost:5000`

3. Upload an image of a plant leaf and click "Upload and Predict" to get the disease detection results.

## Model Training
To facilitate easy access to the project files and dataset, we have made them available via Google Drive. You can access all necessary files using the following link:

[Plant Disease Detection Project Files](https://drive.google.com/drive/folders/16m7syeVUMm9BNUv_Zu6umKzWbEq5cqvk?usp=sharing)

To set up and use these files:

1. Navigate to your Google Drive.
2. Open the "Colab Notebooks" folder.
3. Inside "Colab Notebooks", create a new folder named "plant".
4. Upload the following files to the "Plant" folder:
   - `train.csv`: Training dataset.
   - `test.csv`: Test dataset.
   - `sample_submission.csv`: Sample submission file.
   - `model.h5`: The trained model file.
   - `images`: All the training images
5. Create a subfolder named "images" inside the "Plant" folder.
6. Upload all the training images to the "images" folder.

Ensure you have sufficient storage space in your Google Drive for all these files, especially the images folder which may be large.

Note: The `model.h5` file in this folder is the trained model. If you're starting from scratch, you may not have this file initially. It will be generated after running the training process in the notebook.

The collab notebook includes:
- Data preprocessing
- Model architecture (Xception and DenseNet ensemble)
- Training process
- Model evaluation
- Saving the trained model



## Technologies Used

- Python
- Flask
- TensorFlow
- Pillow (PIL)
- HTML/CSS
- JavaScript

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



