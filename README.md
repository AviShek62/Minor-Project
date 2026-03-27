# Driver Drowsiness Detection System

A deep learning-based system for detecting driver drowsiness in real-time using Convolutional Neural Networks (CNN) and facial recognition. The system analyzes driver eye and facial features to identify signs of drowsiness and can alert the driver to prevent accidents.

## Features

- **Real-time Detection**: Detects drowsiness based on eye closure and facial features
- **CNN-Based Model**: Uses a trained convolutional neural network for accurate classification
- **Multiple Eye States**: Classifies eyes as Open or Closed
- **Facial Expression Analysis**: Detects yawning behavior as an indicator of drowsiness
- **Haar Cascade Integration**: Uses OpenCV's Haar Cascade classifiers for face and eye detection

## Project Structure

```
driver_drowsiness_system_CNN-main/
├── detect_drowsiness.py              # Main detection script
├── driver-drowsiness_notebook.ipynb  # Jupyter notebook with analysis and model training
├── test_tf.py                        # Testing script for the model
├── drowiness_new7.h5                 # Trained model (HDF5 format)
├── drowiness_new7.keras              # Trained model (Keras format)
├── archive/                          # Training data directory
│   └── train/
│       ├── Closed/                   # Images of closed eyes
│       ├── Open/                     # Images of open eyes
│       ├── yawn/                     # Images of yawning
│       └── no_yawn/                  # Images of non-yawning
├── data/                             # Cascade classifier files
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_lefteye_2splits.xml
│   └── haarcascade_righteye_2splits.xml
└── Required_files/                   # Additional cascade files
    ├── haarcascade_frontalface_default.xml
    └── haarcascade.xml
```

## Requirements

- Python 3.7+
- OpenCV (cv2)
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib

Install dependencies using:
```bash
pip install opencv-python tensorflow keras numpy pandas matplotlib
```

## Usage

### Running the Drowsiness Detection

```python
python detect_drowsiness.py
```

This will start the real-time drowsiness detection system using your webcam.

### Testing the Model

```python
python test_tf.py
```

Run this script to test the trained model's performance.

### Running the Notebook

Open the Jupyter notebook for detailed analysis and model training:
```bash
jupyter notebook driver-drowsiness_notebook.ipynb
```

## Model Information

- **Model Type**: Convolutional Neural Network (CNN)
- **Trained Models**: 
  - `drowiness_new7.h5` - HDF5 format
  - `drowiness_new7.keras` - Keras format
- **Classification Categories**:
  - Eye states: Open, Closed
  - Mouth states: Yawn, No Yawn

## Visual Examples

### Eyes Open (Alert State)
![Eyes Open](Required_files/eye%20open%20(1).png)

### Eyes Closed (Drowsiness Alert)
![Eyes Closed](Required_files/eye%20close.png)

## How It Works

1. **Face Detection**: Detects faces in the video stream using Haar Cascade classifier
2. **Eye Detection**: Localizes left and right eyes within detected faces
3. **Feature Extraction**: Extracts relevant features from eye regions
4. **Classification**: Uses the trained CNN model to classify eye states (Open/Closed)
5. **Drowsiness Alert**: Triggers alerts if prolonged eye closure is detected

## Results

The system achieves high accuracy in detecting:
- Eye closure patterns indicative of drowsiness
- Yawning behavior
- Overall fatigue indicators

## Future Enhancements

- Real-time audio/visual alerts
- Integration with vehicle control systems
- Support for multiple camera angles
- Enhanced model with more training data
- Deployment on edge devices (Raspberry Pi, etc.)

## License

This project is provided as-is. See LICENSE file for more details.

## Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests.

## Contact

For questions or suggestions, please reach out through the repository issues.

---

**Note**: This system is intended for research and educational purposes. For production use in vehicles, extensive testing and validation is required.
