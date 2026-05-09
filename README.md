# 🚨 Accident Detection System

A production-grade deep learning system for automatic accident detection from CCTV footage and video files using a fine-tuned VGG16 model with **96% accuracy**.

## Streamlit App Link

[![Open Accident Detection App](https://img.shields.io/badge/Streamlit-App-green?style=for-the-badge)](https://accident-detection-from-cctv-wqc5apbzgp7hbnzueejgra.streamlit.app/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Performance](#performance)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#-api-reference)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Results & Benchmarks](#results--benchmarks)
- [Contributing](#contributing)
- [License](#license)

---

## 📝 Overview

This project develops an intelligent traffic monitoring system that automatically detects road accidents from CCTV camera footage in real-time. The system classifies video frames into two categories:

- **Accident**: Detected vehicle collision/accident
- **Non-Accident**: Normal road traffic

### Use Cases

- 🚗 Real-time traffic accident detection
- 🏢 Smart city surveillance systems
- 🚨 Emergency alert systems
- 📊 AI-powered traffic monitoring
- 🔬 Computer vision research
- 📺 Video anomaly detection
- 🛣️ Intelligent highway monitoring

---

## ✨ Features

### Core Functionality

- ✅ **Real-time Frame Analysis**: Process video frames with high accuracy
- ✅ **Image Analysis**: Single CCTV frame classification
- ✅ **Video Processing**: Frame-by-frame analysis with accident timeline
- ✅ **Confidence Thresholding**: Adjustable sensitivity (0.0 - 1.0)
- ✅ **Accident Timeline**: Timestamp-based incident tracking
- ✅ **Report Generation**: Automated summary reports with statistics
- ✅ **Export Functionality**: Download analysis reports

### Technical Features

- ✅ **Transfer Learning**: Fine-tuned VGG16 backbone
- ✅ **Production-Ready**: Logging, error handling, validation
- ✅ **Modern UI**: Streamlit-based web interface with custom styling
- ✅ **Memory Efficient**: Optimized frame processing
- ✅ **File Validation**: Comprehensive input validation
- ✅ **GPU Support**: TensorFlow GPU acceleration ready

---

## 🎯 Performance

### Model Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 96.0% ⭐ |
| **Precision** | 100.0% |
| **Recall** | 91.0% |
| **Test Loss** | 0.166 |

### Model Comparison

| Model | Accuracy | Loss | Notes |
|-------|----------|------|-------|
| Baseline CNN | 86.0% | 0.322 | Simple CNN architecture |
| Augmented CNN | 84.0% | 0.438 | With data augmentation |
| VGG16 (Transfer) | 88.0% | 0.271 | Pre-trained features |
| **VGG16 (Fine-Tuned)** | **96.0%** | **0.166** | **Final Best Model** ✨ |

### Confusion Matrix (Final Model)

```
                 Predicted
                 Accident  Non-Accident
Actual Accident       43          4
       Non-Accident    0         52
```

---

## 🏗️ Architecture

### Model Stack

- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Input Size**: 224×224 RGB images
- **Fine-tuning Strategy**: Last 16 layers unfrozen
- **Framework**: TensorFlow/Keras
- **Output**: Binary classification (Accident / Non-Accident)

### Application Stack

```
Streamlit UI (Web Interface)
        ↓
Configuration & Constants
        ↓
Image/Video Processors → Prediction Engine
        ↓
Model Manager (VGG16 Loader)
        ↓
TensorFlow Backend
```

### Key Components

1. **ModelManager**: Handles model loading and initialization
2. **ImageProcessor**: Image validation and preprocessing
3. **VideoProcessor**: Frame extraction and batch processing
4. **PredictionEngine**: Core prediction logic
5. **ReportGenerator**: Statistical report creation
6. **UIComponents**: Streamlit UI rendering

---

## 📂 Project Structure

```
accident-detection/
├── app.py                      # Main Streamlit application
├── code.ipynb                  # Jupyter notebook with full analysis
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md                   # This file
```

### Key Files

| File | Purpose |
|------|---------|
| `app.py` | Production Streamlit application (1120 lines) |
| `code.ipynb` | Full ML pipeline, training, and evaluation (63 cells) |
| `final_model.keras` | Pre-trained fine-tuned VGG16 model |

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- OpenCV 4.x
- Streamlit 1.x
- 2GB+ free disk space (for model)

### Step 1: Clone or Download

```bash
# Download the project files
# Place app.py and final_model.keras in the same directory
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Requirements File

```txt
streamlit>=1.28.0
tensorflow>=2.13.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
gdown
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

---

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will launch at: `http://localhost:8501`

### Web Interface Guide

#### 1. **Image Analysis Mode** 🖼️

1. Select "🖼️ Image Analysis" from the sidebar
2. Upload a CCTV frame (JPG, PNG, BMP)
3. Adjust confidence threshold if needed
4. View prediction results:
   - Original image display
   - Classification (Accident/Safe)
   - Confidence percentage
   - Probability breakdown

#### 2. **Video Analysis Mode** 📹

1. Select "📹 Video Analysis" from the sidebar
2. Upload a video file (MP4, AVI, MOV, MKV, FLV)
3. System processes frame-by-frame
4. Review results:
   - Total frames analyzed
   - Accident timeline with timestamps
   - Detection statistics
   - Average confidence metrics

#### 3. **Confidence Threshold** ⚙️

Adjust in sidebar (default: 0.5):
- **Lower** (0.3): More sensitive, catches more accidents (higher false positives)
- **Default** (0.5): Balanced approach
- **Higher** (0.8): More conservative, only clear accidents (fewer false negatives)

#### 4. **Report Export** 📥

- Click "Generate Summary Report"
- Review preview in text area
- Download as `.txt` file with timestamp

---

## 🔌 API Reference

### Core Classes

#### ModelManager

```python
class ModelManager:
    def load_model(path: str) -> bool
        """Load VGG16 model from file"""
    
    def is_loaded() -> bool
        """Check if model is ready"""
    
    def get_model() -> tf.keras.Model
        """Return loaded model"""
```

#### PredictionEngine

```python
class PredictionEngine:
    def predict_image(image: PIL.Image) -> Tuple[bool, float, float, float]
        """
        Predict if image contains accident
        Returns: (is_accident, confidence, accident_prob, non_accident_prob)
        """
    
    def predict_batch(images: List[PIL.Image]) -> List[Tuple]
        """Batch prediction for multiple images"""
```

#### VideoProcessor

```python
class VideoProcessor:
    def process(video_path: str, threshold: float) -> Dict
        """
        Process entire video file
        Returns: {
            'total_frames': int,
            'duration': float,
            'fps': float,
            'accident_frames': List[Dict],
            'all_results': List[Dict]
        }
        """
```

#### ReportGenerator

```python
class ReportGenerator:
    def generate_text_report(video_results: Dict, threshold: float) -> str
        """Generate plain text summary report"""
```

### Configuration

```python
class Config:
    MODEL_PATH = "final_model.keras"
    TARGET_IMAGE_SIZE = (224, 224)
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    VIDEO_PROCESSING_INTERVAL = 2  # frames/sec
    ALLOWED_IMAGE_TYPES = ("jpg", "jpeg", "png", "bmp")
    ALLOWED_VIDEO_TYPES = ("mp4", "avi", "mov", "mkv", "flv")
    MAX_FILE_SIZE_MB = 500
    MODEL_ACCURACY = 0.96
```

---

## 🧠 Model Details

### Architecture

```
Input: 224×224 RGB Image
        ↓
VGG16 Backbone (Pre-trained ImageNet weights)
        ↓
Flatten
        ↓
Dense Layer (256 units, ReLU)
        ↓
Dropout (0.5)
        ↓
Output Layer (1 units, Sigmoid)
        ↓
Binary Classification: [Non-Accident probability, Accident probability]
```

### Training Strategy

1. **Phase 1**: Feature extraction (VGG16 frozen)
   - Quick training on pre-learned features
   - Achieves 88% accuracy

2. **Phase 2**: Fine-tuning (Last 16 layers unfrozen)
   - Gradual learning rate adjustment
   - Trains on last convolutional blocks
   - Achieves 96% accuracy

### Input Preprocessing

- Resize to 224×224
- Convert to RGB
- Normalize pixel values [0, 1]
- No aggressive augmentation during inference

### Training Data

- **Total Images**: ~800 CCTV frames
- **Training Set**: ~600 images (75%)
- **Validation Set**: ~100 images (12.5%)
- **Test Set**: ~100 images (12.5%)
- **Class Distribution**:
  - Accident: 46.8%
  - Non-Accident: 53.2%
  - Minor imbalance (~6%) - acceptable

---

## 📊 Results & Benchmarks

### Performance Comparison Across Models

```
Accuracy Progression:
86% (Baseline CNN)
   ↓
84% (Augmented CNN)
   ↓
88% (VGG16 Transfer)
   ↓
96% (VGG16 Fine-Tuned) ✨ FINAL
```

### Key Insights

1. **Transfer Learning Impact**: +8% improvement over baseline
2. **Fine-tuning Benefits**: +8% additional improvement
3. **Precision Excellence**: 100% precision prevents false alarms
4. **Recall Trade-off**: 91% recall catches most real accidents
5. **Data Imbalance**: Minimal impact with only 6% class difference

### Threshold Analysis

- **Threshold 0.5**: 96% accuracy, balanced approach
- **Adjusted Threshold**: Maintains 95% accuracy with improved sensitivity
- **Confusion Matrix**: Only 5 misclassifications per ~100 frames

### Inference Time

- **Per-frame**: ~10-20ms (GPU)
- **Per-frame**: ~50-100ms (CPU)
- **Video processing**: ~2 frames/second (adaptive)

---

## 🛠️ Development & Training

### Jupyter Notebook (`code.ipynb`)

The notebook includes:

- **EDA**: Dataset exploration and visualization
- **Preprocessing**: Image loading, normalization, augmentation
- **Model 1**: Baseline CNN (86% accuracy)
- **Model 2**: Augmented CNN (84% accuracy)
- **Model 3**: VGG16 Transfer Learning (88% accuracy)
- **Model 4**: VGG16 Fine-tuning (96% accuracy) ✨
- **Evaluation**: Confusion matrices, ROC curves, reports
- **Threshold Tuning**: Optimizing decision boundary
- **Comparison**: Side-by-side model analysis

### Running Training

```bash
# Open and execute in Jupyter
jupyter notebook code.ipynb

# Or run all cells
jupyter nbconvert --to notebook --execute code.ipynb
```

### Dataset

- **Source**: Accident Detection From CCTV Footage Dataset
- **Creator**: Charan Kumar
- **Images**: Real CCTV frames from traffic scenes
- **Categories**: 2 classes (Accident, Non-Accident)
- **Download**: [Kaggle Dataset](https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage/data)

---

## 📈 Logging & Monitoring

### Log Output

Application logs are saved to `accident_detection.log`:

```
2024-01-15 10:23:45 - __main__ - INFO - Image prediction completed
2024-01-15 10:24:12 - __main__ - INFO - Video processing started
2024-01-15 10:25:33 - __main__ - ERROR - Invalid file format
```

### Log Levels

- `DEBUG`: Detailed debugging information
- `INFO`: General informational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical system errors

---

## 🔒 Error Handling

The application includes comprehensive error handling for:

- ❌ Invalid image formats
- ❌ Unsupported video types
- ❌ File size exceeded
- ❌ Model loading failures
- ❌ GPU/memory errors
- ❌ Corrupted files
- ❌ Network timeouts

All errors are logged and displayed to the user with helpful messages.

---

## ⚙️ Configuration Options

### Adjustable Parameters

```python
# Model Settings
TARGET_IMAGE_SIZE = (224, 224)           # Input resolution
DEFAULT_CONFIDENCE_THRESHOLD = 0.5       # Decision boundary

# Video Processing
VIDEO_PROCESSING_INTERVAL = 2            # Frames analyzed per second
VIDEO_DISPLAY_INTERVAL = 10              # Preview update frequency
MAX_FILE_SIZE_MB = 500                   # Maximum video size

# File Types
ALLOWED_IMAGE_TYPES = ("jpg", "jpeg", "png", "bmp")
ALLOWED_VIDEO_TYPES = ("mp4", "avi", "mov", "mkv", "flv")
```

---

## 🚨 Limitations & Considerations

1. **Model Limitations**:
   - Trained on specific CCTV characteristics
   - May not generalize to all camera angles/qualities
   - Requires clear visibility of vehicles

2. **Performance**:
   - Video processing is sequential (not parallel)
   - Large files (>500MB) may require longer processing
   - CPU inference is slower than GPU

3. **Data**:
   - Relatively small training set (~800 images)
   - Limited geographic/weather diversity
   - Day-time dominant training data

4. **Deployment**:
   - Streamlit designed for single-user or small-scale use
   - For production: consider Flask/FastAPI wrapper
   - GPU recommended for real-time processing

---

## 📖 Usage Examples

### Example 1: Analyze Single Image

```python
from PIL import Image
from app import PredictionEngine, ModelManager

# Load model
model_mgr = ModelManager()
model_mgr.load_model("final_model.keras")

# Create prediction engine
engine = PredictionEngine(model_mgr)

# Load and predict
image = Image.open("accident_frame.jpg")
is_accident, confidence, acc_prob, non_acc_prob = engine.predict_image(image)

print(f"Accident detected: {is_accident}")
print(f"Confidence: {confidence:.1%}")
```

### Example 2: Process Video Programmatically

```python
from app import VideoProcessor, ModelManager, PredictionEngine

# Initialize
model_mgr = ModelManager()
model_mgr.load_model("final_model.keras")
engine = PredictionEngine(model_mgr)
processor = VideoProcessor(model_mgr, engine)

# Process video
results = processor.process("traffic_video.mp4", threshold=0.5)

# Access results
print(f"Total frames: {results['total_frames']}")
print(f"Accidents detected: {len(results['accident_frames'])}")
for accident in results['accident_frames']:
    print(f"  - Frame {accident['frame_number']} at {accident['timestamp']:.1f}s")
```

### Example 3: Generate Report

```python
from app import ReportGenerator

report_gen = ReportGenerator()
report = report_gen.generate_text_report(video_results, threshold=0.5)

# Save report
with open("analysis_report.txt", "w") as f:
    f.write(report)
```

---

## 🔧 Troubleshooting

### Issue: Model not found

```
Error: Model failed to load
```

**Solution**: Ensure `final_model.keras` is in the same directory as `app.py`

### Issue: Out of memory

```
Error: Could not allocate X GB of GPU memory
```

**Solution**: Reduce `VIDEO_PROCESSING_INTERVAL` or use CPU (`os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`)

### Issue: Slow video processing

```
Video processing taking too long
```

**Solution**: Increase `VIDEO_PROCESSING_INTERVAL` (process fewer frames) or use GPU

### Issue: Port already in use

```
Error: Port 8501 is already in use
```

**Solution**: Run on different port: `streamlit run app.py --server.port 8502`

---

## 📚 References

- [TensorFlow VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [ImageNet Dataset](http://www.image-net.org/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Multi-GPU support
- [ ] REST API wrapper
- [ ] Mobile app integration
- [ ] Real-time streaming support
- [ ] Improved data augmentation
- [ ] Model optimization for edge devices
- [ ] Spatial localization of accidents

---

## 📞 Support

For issues or questions:

1. Check the **Troubleshooting** section
2. Review application logs in `accident_detection.log`
3. Verify all dependencies are installed
4. Ensure model file exists and is valid

---

## ✅ Checklist for Deployment

- [ ] Model file (`final_model.keras`) present
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Sufficient disk space (2GB+)
- [ ] Python 3.8+ installed
- [ ] GPU drivers installed (optional, for faster processing)
- [ ] File permissions correctly set
- [ ] Logs directory writable
- [ ] Network configured (if remote deployment)

---

## 🎓 Educational Value

This project demonstrates:

- ✅ Transfer learning techniques
- ✅ Fine-tuning pre-trained models
- ✅ Binary image classification
- ✅ Video frame processing
- ✅ Deep learning with TensorFlow/Keras
- ✅ Streamlit web application development
- ✅ Production ML pipeline design
- ✅ Model evaluation and comparison
- ✅ Threshold optimization
- ✅ Real-world computer vision applications

---

**Last Updated**: January 2024  
**Version**: 1.0  
**Status**: Production Ready ✨

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| Model Accuracy | 96% |
| Precision | 100% |
| Recall | 91% |
| Total Code Lines | 1,100+ |
| Jupyter Cells | 63 |
| Supported Formats | 4 image + 5 video |
| Max File Size | 500 MB |
| UI Components | 8 custom classes |

---

*Built with ❤️ using TensorFlow, Keras, and Streamlit*
