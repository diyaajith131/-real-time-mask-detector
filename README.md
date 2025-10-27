# 🛡️ Face Mask Detection with Live Alert System

A real-time face mask detection system built with Python, OpenCV, TensorFlow, and Flask. This project uses transfer learning with MobileNetV2 to detect whether people are wearing face masks in real-time through a webcam feed.

## 🚀 Features

- **Real-time Detection**: Live webcam feed with instant mask detection
- **AI-Powered**: Uses MobileNetV2 transfer learning for accurate detection
- **Web Interface**: Beautiful, responsive web interface for monitoring
- **Confidence Scores**: Shows prediction confidence for each detection
- **Alert System**: Visual alerts with color-coded bounding boxes
- **Easy Setup**: Simple installation and configuration

## 📁 Project Structure

```
Diya_inter/
├── data/
│   ├── with_mask/          # Training images of people wearing masks
│   └── without_mask/       # Training images of people not wearing masks
├── models/
│   └── face_mask_detector.h5  # Trained model (generated after training)
├── templates/
│   └── index.html         # Web interface template
├── train.py              # Model training script
├── app.py                # Flask web server
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- At least 4GB RAM (8GB recommended for training)

### Step 1: Clone and Setup

```bash
# Navigate to your project directory
cd /Users/sci_coderamicia/Diya_inter

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Training Data

1. **Collect Images**: Gather images of people wearing and not wearing masks
2. **Organize Data**: 
   - Place images of people **with masks** in `data/with_mask/`
   - Place images of people **without masks** in `data/without_mask/`
3. **Recommended**: At least 100-200 images per class for good results

### Step 3: Train the Model

```bash
python train.py
```

This will:
- Load and preprocess your training data
- Create a MobileNetV2-based model with transfer learning
- Train the model for 20 epochs with data augmentation
- Save the trained model to `models/face_mask_detector.h5`
- Display training progress and final accuracy

### Step 4: Run the Live Detection System

```bash
python app.py
```

Then open your browser and go to: `http://localhost:5000`

## 🎯 Usage

### Training the Model

1. **Prepare your dataset**:
   - Collect images of people with and without masks
   - Ensure good variety in lighting, angles, and backgrounds
   - Aim for balanced classes (similar number of images per class)

2. **Run training**:
   ```bash
   python train.py
   ```

3. **Monitor training**:
   - Watch the accuracy and loss curves
   - Training typically takes 10-30 minutes depending on your hardware
   - The model will be automatically saved when training completes

### Running Live Detection

1. **Start the server**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   - Open `http://localhost:5000` in your browser
   - Allow camera permissions when prompted
   - The system will start detecting faces and masks in real-time

3. **Understanding the output**:
   - **Green box + "MASK"**: Person is wearing a mask
   - **Red box + "NO MASK"**: Person is not wearing a mask
   - Confidence scores are displayed with each detection

## 🔧 Configuration

### Model Parameters

You can modify these parameters in `train.py`:

```python
# Training parameters
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

# Data augmentation
rotation_range = 20
width_shift_range = 0.2
height_shift_range = 0.2
```

### Detection Parameters

Modify detection settings in `app.py`:

```python
# Face detection parameters
scaleFactor = 1.1
minNeighbors = 5
minSize = (30, 30)
```

## 📊 Performance Tips

### For Better Accuracy

1. **Quality Dataset**: Use high-quality, diverse images
2. **Balanced Classes**: Ensure equal representation of both classes
3. **Good Lighting**: Train with various lighting conditions
4. **Multiple Angles**: Include different face angles and positions

### For Better Performance

1. **GPU Acceleration**: Install `tensorflow-gpu` for faster training
2. **Reduce Resolution**: Lower input resolution for faster inference
3. **Batch Processing**: Process multiple frames together
4. **Model Optimization**: Use TensorFlow Lite for mobile deployment

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**:
   ```bash
   pip install opencv-python
   ```

2. **"Model not found"**:
   - Make sure you've run `train.py` first
   - Check that `models/face_mask_detector.h5` exists

3. **Camera not working**:
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Ensure no other applications are using the camera

4. **Low accuracy**:
   - Add more training data
   - Improve data quality and diversity
   - Increase training epochs
   - Check data balance between classes

### Performance Issues

1. **Slow inference**:
   - Reduce input image size
   - Use GPU acceleration
   - Optimize model architecture

2. **High memory usage**:
   - Reduce batch size
   - Use smaller model
   - Close other applications

## 🔒 Security Notes

- This system is for demonstration purposes
- Ensure camera permissions are properly managed
- Consider data privacy when collecting training images
- The model runs locally - no data is sent to external servers

## 📈 Future Enhancements

- **Multi-person detection**: Detect multiple people simultaneously
- **Mask type classification**: Distinguish between different mask types
- **Social distancing**: Add distance measurement between people
- **Mobile deployment**: Convert to TensorFlow Lite for mobile apps
- **Database integration**: Store detection logs and statistics
- **Alert system**: Send notifications when violations are detected

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- TensorFlow team for the excellent deep learning framework
- OpenCV community for computer vision tools
- Flask team for the web framework
- Contributors to the MobileNetV2 architecture


