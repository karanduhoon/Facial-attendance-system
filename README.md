

---

# Attendance System Using Machine Learning

## Overview

The Attendance System Using Machine Learning automates attendance tracking by detecting and recognizing individuals in real-time using computer vision techniques and deep learning models. This system integrates a trained facial recognition model with a Django web application to track attendance automatically.

## Problem Statement

Traditional attendance systems are often inefficient, relying on manual or biometric methods. This system automates attendance tracking by detecting individuals and logging their attendance based on real-time video feeds, leveraging machine learning and facial recognition.

## Features

- **Real-time Attendance Tracking**: Automatically marks attendance using a camera feed.
- **YOLO Object Detection**: Detects human figures in video frames.
- **Facial Recognition**: Identifies known individuals using pre-registered facial data.
- **Attendance Logging**: Saves attendance records with timestamps in a database.
- **Data Augmentation**: Enhances model accuracy through image transformations.

## Project Structure

The project consists of two main components:

1. **ML Model Training** (`MLprojFinal.ipynb`): This Jupyter Notebook is where the machine learning model for facial recognition is trained using various techniques.
2. **Django Web Application** (`attendance_app`): This folder contains the Django project that integrates the trained model and performs real-time attendance tracking.

### File Structure

```plaintext
attendance-app/
│
├── MLprojFinal.ipynb                # Jupyter Notebook for training the model
└── facial_attendance/               # Django project folder
    ├── attendance_app/              # Django app folder
    ├── manage.py                    # Django management script
    ├── settings.py                  # Django settings configuration
    └── ...
```

## Installation

### Prerequisites

- Python 3.x
- Django
- OpenCV
- TensorFlow or PyTorch (for the deep learning model)
- `yolov8` model (or any compatible YOLO model)
- Scikit-learn, NumPy, Pandas, and other necessary libraries

### Setup

#### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/attendance-system.git
cd attendance-system
```

#### 2. Install dependencies for the ML model training:
In the `MLprojFinal.ipynb` Jupyter notebook, the following libraries are used:
```bash
pip install opencv-python tensorflow scikit-learn pandas numpy
```

#### 3. Install Django and related dependencies for the web application:
```bash
pip install django opencv-python
```

#### 4. Download the pre-trained YOLO model and save it in the appropriate folder for object detection.

#### 5. Set up the Django project:
1. Navigate to the `attendance_app` directory.
2. Run the Django development server:
   ```bash
   python manage.py runserver
   ```

#### 6. Access the web app:
Once the server is running, open a web browser and visit `http://localhost:8000` to interact with the facial attendance system.

## Usage

1. **Train the model**: 
   - Open `MLprojFinal.ipynb` and run the cells to train the facial recognition model. The notebook trains the model to recognize faces using your dataset. The trained model will be saved for use in the Django app.

2. **Integrate the model with Django**:
   - After training, the model is used within the Django web application (`attendance_app`) for real-time face detection and attendance tracking.
   - The Django app uses OpenCV to capture video, detect faces, and compare them with the registered profiles in the system.

3. **Real-time Attendance**:
   - The web application will display real-time video feeds from the camera, and the attendance of recognized individuals will be logged automatically.

4. **Attendance Data**:
   - The attendance data, including the timestamp and the names of recognized individuals, will be saved in the database and can be viewed in the web application.

## Methodology

### Object Detection with YOLO

The **YOLO (You Only Look Once)** model is used for real-time object detection. It detects human figures in video frames and provides bounding boxes and class probabilities for each detected individual.

### Facial Recognition

The system uses **transfer learning** with pre-trained models to recognize faces. Facial embeddings (using models like **FaceNet**) are compared with stored profiles to identify individuals.

### Data Augmentation

Data augmentation techniques, such as scaling, rotation, and translation, are applied to enhance the training data, improving model accuracy.

## Performance Evaluation

- **Detection Accuracy**: 78.4% mAP50, 55.7% mAP50-95
- **Precision**: 73.3%
- **Recall**: 62.5%
- **Inference Speed**: 2.1 ms per frame (near real-time performance)

## Challenges

- **Lighting Conditions**: The model’s performance may degrade in low-light environments.
- **Crowded Environments**: The system may struggle with detecting individuals when they are too close to each other.
- **Occlusion**: Partial obstruction of individuals may reduce detection accuracy.

## Future Improvements

- **Face Recognition**: Enhance recognition accuracy by integrating more advanced face recognition models.
- **Scalability**: Improve the system to handle larger datasets and complex environments.
- **Robustness to Variability**: Address challenges such as lighting changes, masks, and occlusions.

## Conclusion

This project demonstrates how machine learning, object detection, and facial recognition can be used to automate attendance tracking. The combination of a trained model and a Django web application provides a robust solution for real-time attendance management.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 Object Detection Model
- Kaggle Dataset for Facial Images
- OpenCV for Video Processing
- Django for Web Application Framework

---
