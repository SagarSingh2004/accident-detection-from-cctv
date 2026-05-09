"""
🚨 Accident Detection System - Image & Video Analysis
======================================================
Production-grade accident detection from CCTV footage and video files
Using Fine-Tuned VGG16 model (96% accuracy)

Run: streamlit run app.py
"""

import streamlit as st
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any
import tempfile
import gdown

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Config:
    """Application configuration"""
    # Model settings
    MODEL_PATH = "final_model.keras"
    # Direct download link for gdown
    MODEL_URL = "https://drive.google.com/uc?id=1U6VMR1v1f84yyAJqJfqOnskKQFdi1RBN"

    TARGET_IMAGE_SIZE = (224, 224)
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    
    # Video processing
    VIDEO_PROCESSING_INTERVAL = 2  # Process ~2 frames per second
    VIDEO_DISPLAY_INTERVAL = 10
    VIDEO_PROGRESS_UPDATE_INTERVAL = 50
    
    # File validation
    ALLOWED_IMAGE_TYPES = ("jpg", "jpeg", "png", "bmp")
    ALLOWED_VIDEO_TYPES = ("mp4", "avi", "mov", "mkv", "flv")
    MAX_FILE_SIZE_MB = 500
    
    # UI settings
    PAGE_TITLE = "🚨 Accident Detection"
    PAGE_ICON = "🚗"
    LAYOUT = "wide"
    
    # Model info
    MODEL_ACCURACY = 0.96
    MODEL_PRECISION = 1.0
    MODEL_RECALL = 0.91
    
    # Logging
    LOG_LEVEL = logging.INFO

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging for the application"""
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('accident_detection.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# UI STYLING
# ============================================================================

def apply_custom_css():
    """Modern colorful UI"""

    st.markdown("""
    <style>

    /* ===== Main App Background ===== */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #111827 25%,
            #1e293b 50%,
            #0f172a 100%
        );
        color: white;
    }

    /* ===== Hide Streamlit Branding ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ===== Main Header ===== */
    .main-header {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(
            90deg,
            #38bdf8,
            #818cf8,
            #c084fc
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }

    /* ===== Glass Containers ===== */
    .block-container {
        padding-top: 2rem;
    }

    div[data-testid="stVerticalBlock"] > div {
        border-radius: 20px;
    }

    /* ===== Prediction Boxes ===== */

        .prediction-accident, .prediction-safe {
            padding: 18px 10px;
            border-radius: 18px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 700;
            border: 1px solid rgba(255,255,255,0.18);
            margin-bottom: 0.5rem;
            box-shadow: 0 4px 18px rgba(0,0,0,0.18);
            max-width: 420px;
            margin-left: auto;
            margin-right: auto;
        }
        .prediction-accident {
            background: linear-gradient(135deg, #ff0055cc, #ff5555cc);
            color: #fff;
            animation: pulse 2s infinite;
        }
        .prediction-safe {
            background: linear-gradient(135deg, #10b981cc, #22c55ecc);
            color: #fff;
        }

    /* ===== Card Styling ===== */

    .confidence-box,
    .stats-box,
    .video-progress {
        background: #192132;
        border-radius: 18px;
        padding: 18px 16px 10px 16px;
        border: 1px solid #334155;
        box-shadow: 0 4px 16px rgba(0,0,0,0.18);
        color: #f1f5f9;
        margin-top: 12px;
        margin-bottom: 10px;
    }
    .confidence-box .stMetric, .confidence-box .stMetricLabel, .confidence-box .stMetricValue, .confidence-box .stMetricDelta {
        color: #f1f5f9 !important;
        font-weight: 600;
    }
    .confidence-box .stMetricValue {
        font-size: 1.3rem !important;
    }
    .confidence-box .stMetricLabel {
        font-size: 1rem !important;
    }

    /* ===== Sidebar ===== */

    section[data-testid="stSidebar"] {

        background: linear-gradient(
            180deg,
            #111827,
            #1e293b
        );

        border-right: 1px solid rgba(255,255,255,0.1);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* ===== Buttons ===== */

    .stButton>button {

        width: 100%;

        border-radius: 15px;

        border: none;

        background: linear-gradient(
            90deg,
            #06b6d4,
            #3b82f6
        );

        color: white;

        font-weight: 700;

        padding: 12px 20px;

        transition: 0.3s;

        box-shadow:
            0 4px 15px rgba(59,130,246,0.4);
    }

    .stButton>button:hover {

        transform: translateY(-2px);

        box-shadow:
            0 8px 25px rgba(59,130,246,0.6);
    }

    /* ===== Upload Box ===== */

    .stFileUploader {

        background: rgba(255,255,255,0.05);

        border-radius: 20px;

        padding: 20px;

        border: 2px dashed rgba(255,255,255,0.2);
    }

    /* ===== Progress Bar ===== */

    .stProgress > div > div > div > div {

        background: linear-gradient(
            90deg,
            #06b6d4,
            #8b5cf6
        );
    }

    /* ===== Metrics ===== */

    div[data-testid="metric-container"] {

        background: rgba(255,255,255,0.08);

        border-radius: 18px;

        padding: 15px;

        border: 1px solid rgba(255,255,255,0.1);

        box-shadow:
            0 6px 20px rgba(0,0,0,0.25);
    }

    /* ===== Animation ===== */

    @keyframes pulse {

        0% {
            transform: scale(1);
        }

        50% {
            transform: scale(1.02);
        }

        100% {
            transform: scale(1);
        }
    }

    /* ===== Dataframe ===== */

    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
    }

    /* ===== Mobile ===== */

    @media (max-width: 768px) {

        .main-header {
            font-size: 2rem;
        }

        .prediction-accident,
        .prediction-safe {

            font-size: 1.3rem;
            padding: 25px;
        }
    }

    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    """Handles model loading and caching"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _download_model(self) -> None:
        """Download model from Google Drive if not exists"""

        try:
            if not Path(self.model_path).exists():

                st.sidebar.info("⬇️ Downloading model...")

                logger.info("Downloading model from Google Drive")

                gdown.download(
                    Config.MODEL_URL,
                    self.model_path,
                    quiet=False
                )

                logger.info("✅ Model downloaded successfully")

        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}", exc_info=True)
            st.sidebar.error(f"❌ Model download failed: {str(e)}")
    
    def _load_model(self) -> None:
        """Load the pre-trained model with error handling"""

        try:
            # Download model if missing
            self._download_model()

            if not Path(self.model_path).exists():

                logger.error(f"Model file not found: {self.model_path}")

                st.error(f"❌ Model file not found: {self.model_path}")

                return

            st.sidebar.info("🧠 Loading model...")

            self.model = load_model(
                self.model_path,
                compile=False
            )

            logger.info(f"✅ Model loaded successfully: {self.model_path}")

            st.sidebar.success("✅ Model loaded successfully")

        except Exception as e:

            logger.error(
                f"Failed to load model: {str(e)}",
                exc_info=True
            )

            st.sidebar.error(
                f"❌ Error loading model: {str(e)}"
            )
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model(self):
        """Get the loaded model"""

        if not self.is_loaded():

            raise RuntimeError(
                "Model not loaded. Check model path and try again."
            )

        return self.model


@st.cache_resource
def get_model_manager() -> ModelManager:
    """Load and cache the model manager"""

    return ModelManager(Config.MODEL_PATH)

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

class ImageProcessor:
    """Handles image preprocessing and prediction"""
    
    @staticmethod
    def validate_image(image: Image.Image) -> bool:
        """Validate image format and properties"""
        try:
            if image.mode not in ('RGB', 'RGBA', 'L'):
                logger.warning(f"Unexpected image mode: {image.mode}")
            return True
        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            return False
    
    @staticmethod
    def preprocess(
        img_input: Any,
        target_size: Tuple[int, int] = Config.TARGET_IMAGE_SIZE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for model prediction
        
        Args:
            img_input: PIL Image or numpy array
            target_size: Target image size for model
            
        Returns:
            Tuple of (batch array, processed array)
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(img_input, Image.Image):
                img_array = np.array(img_input.convert('RGB'))
            else:
                img_array = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            img_resized = cv2.resize(img_array, target_size)
            
            # Normalize to [0, 1]
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            logger.debug(f"Image preprocessed: shape={img_batch.shape}")
            return img_batch, img_resized
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}", exc_info=True)
            raise

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

class PredictionEngine:
    """Handles model predictions and result formatting"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.image_processor = ImageProcessor()
    
    def predict_image(
        self,
        image_input: Any
    ) -> Tuple[Optional[bool], Optional[float], float, float]:
        """
        Predict if image contains accident
        
        Args:
            image_input: PIL Image or numpy array
            
        Returns:
            Tuple of (is_accident, confidence, accident_prob, non_accident_prob)
        """
        try:
            # Preprocess image
            img_batch, _ = self.image_processor.preprocess(image_input)
            
            # Get prediction
            model = self.model_manager.get_model()
            prediction = model.predict(img_batch, verbose=0)
            
            # Parse probabilities
            non_accident_prob = float(prediction[0][0])
            accident_prob = 1.0 - non_accident_prob
            
            # Determine classification
            is_accident = non_accident_prob <= 0.5
            confidence = accident_prob if is_accident else non_accident_prob
            
            logger.info(
                f"Prediction: accident={is_accident}, confidence={confidence:.3f}"
            )
            
            return is_accident, confidence, accident_prob, non_accident_prob
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            st.error(f"❌ Prediction error: {str(e)}")
            return None, None, None, None

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

class VideoProcessor:
    """Handles video processing and frame-by-frame analysis"""
    
    def __init__(self, model_manager: ModelManager, prediction_engine: PredictionEngine):
        self.model_manager = model_manager
        self.prediction_engine = prediction_engine
    
    def get_video_properties(self, video_path: str) -> Dict[str, Any]:
        """Extract video properties"""
        try:
            cap = cv2.VideoCapture(video_path)
            props = {
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            cap.release()
            logger.info(f"Video properties: {props}")
            return props
        except Exception as e:
            logger.error(f"Failed to get video properties: {str(e)}")
            raise
    
    def process(
        self,
        video_path: str,
        confidence_threshold: float = Config.DEFAULT_CONFIDENCE_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Process video frame by frame and detect accidents
        
        Args:
            video_path: Path to video file
            confidence_threshold: Minimum confidence to trigger alert
            
        Returns:
            Dictionary containing analysis results
        """
        cap = cv2.VideoCapture(video_path)
        
        try:
            # Get video properties
            props = self.get_video_properties(video_path)
            total_frames = props['total_frames']
            fps = props['fps']
            
            results = []
            accident_frames = []
            frame_count = 0
            
            # UI placeholders
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_display = st.empty()
            
            # Determine frame processing interval
            process_interval = max(1, int(fps / Config.VIDEO_PROCESSING_INTERVAL))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process selected frames
                if frame_count % process_interval == 0:
                    is_accident, confidence, accident_prob, non_accident_prob = \
                        self.prediction_engine.predict_image(frame)
                    
                    if is_accident and confidence and confidence > confidence_threshold:
                        accident_frames.append({
                            'frame_number': frame_count,
                            'timestamp': frame_count / fps,
                            'confidence': confidence,
                            'accident_prob': accident_prob
                        })
                    
                    results.append({
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps,
                        'is_accident': is_accident,
                        'confidence': confidence,
                        'accident_prob': accident_prob,
                        'non_accident_prob': non_accident_prob
                    })
                    
                    # Update UI
                    progress = frame_count / total_frames
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(
                        f"Processing... {frame_count}/{total_frames} frames | "
                        f"Accidents detected: {len(accident_frames)}"
                    )
                    
                    # Display frame
                    if frame_count % (process_interval * Config.VIDEO_DISPLAY_INTERVAL) == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_display.image(
                            frame_rgb,
                            use_container_width=True,
                            caption=f"Frame {frame_count} - "
                                   f"{'🚨 ACCIDENT' if is_accident else '✅ SAFE'}"
                        )
            
            logger.info(
                f"Video processing complete: {len(accident_frames)} accidents detected"
            )
            
            return {
                'total_frames': total_frames,
                'fps': fps,
                'duration': total_frames / fps,
                'width': props['width'],
                'height': props['height'],
                'all_results': results,
                'accident_frames': accident_frames
            }
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}", exc_info=True)
            st.error(f"❌ Video processing error: {str(e)}")
            return None
        
        finally:
            cap.release()

# ============================================================================
# REPORT GENERATION
# ============================================================================

class ReportGenerator:
    """Generate analysis reports"""
    
    @staticmethod
    def generate_text_report(video_results: Dict[str, Any], threshold: float) -> str:
        """Generate text report from video analysis"""
        try:
            accident_frames = video_results['accident_frames']
            all_results = video_results['all_results']
            
            total_analyzed = len(all_results)
            accident_count = sum(1 for r in all_results if r['is_accident'])
            
            report = f"""
ACCIDENT DETECTION VIDEO ANALYSIS REPORT
{'='*70}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

VIDEO INFORMATION
{'-'*70}
  Total Frames:      {video_results['total_frames']}
  Duration:          {video_results['duration']:.1f}s ({video_results['duration']/60:.1f}min)
  FPS:               {video_results['fps']:.0f}
  Resolution:        {video_results['width']}x{video_results['height']}

ANALYSIS RESULTS
{'-'*70}
  Frames Analyzed:   {total_analyzed}
  Accident Frames:   {accident_count} ({accident_count/total_analyzed*100:.1f}%)
  Safe Frames:       {total_analyzed-accident_count} ({(total_analyzed-accident_count)/total_analyzed*100:.1f}%)
  Confidence Threshold: {threshold:.0%}

DETECTED INCIDENTS
{'-'*70}
"""
            
            if accident_frames:
                for idx, acc in enumerate(accident_frames, 1):
                    minutes = int(acc['timestamp'] // 60)
                    seconds = int(acc['timestamp'] % 60)
                    report += f"\nIncident {idx}:\n"
                    report += f"  Timestamp:  {minutes:02d}:{seconds:02d}\n"
                    report += f"  Frame:      {acc['frame_number']}\n"
                    report += f"  Confidence: {acc['confidence']:.1%}\n"
            else:
                report += "\nNo accidents detected in this video.\n"
            
            report += f"\n{'='*70}\n"
            
            logger.info("Report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}", exc_info=True)
            return "Error generating report"

# ============================================================================
# UI COMPONENTS
# ============================================================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_header():
        """Render application header"""
        st.markdown(
            '<h1 class="main-header">🚨 Accident Detection System</h1>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="subtitle">Image & Video Analysis with Confidence Scores</p>',
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_sidebar_settings() -> Tuple[str, float]:
        """Render sidebar settings and return mode and threshold"""
        with st.sidebar:
            st.header("⚙️ Settings")
            
            mode = st.radio(
                "Select Mode",
                ["🖼️ Image Analysis", "📹 Video Analysis"],
                label_visibility="collapsed"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.3,
                max_value=0.9,
                value=Config.DEFAULT_CONFIDENCE_THRESHOLD,
                step=0.05,
                help="Only alert if confidence exceeds this threshold"
            )
            
            st.divider()
            st.info(f"""
            **Model Performance:**
            - Accuracy: {Config.MODEL_ACCURACY:.0%}
            - Precision: {Config.MODEL_PRECISION:.0%}
            - Recall: {Config.MODEL_RECALL:.0%}
            - Input: {Config.TARGET_IMAGE_SIZE[0]}×{Config.TARGET_IMAGE_SIZE[1]}
            """)
            
            return mode, confidence_threshold
    
    @staticmethod
    def render_prediction_result(
        is_accident: bool,
        confidence: float,
        accident_prob: float,
        non_accident_prob: float
    ):
        """Render prediction results"""
        # Prediction box
        if is_accident:
            st.markdown(
                '<div class="prediction-accident">🚨 ACCIDENT DETECTED</div>',
                unsafe_allow_html=True
            )
            prediction_label = "ACCIDENT"
            color_code = "🔴"
        else:
            st.markdown(
                '<div class="prediction-safe">✅ NO ACCIDENT</div>',
                unsafe_allow_html=True
            )
            prediction_label = "NON ACCIDENT"
            color_code = "🟢"
        
        # Confidence scores
        st.markdown('<div class="confidence-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Confidence Scores")
        col_acc, col_nonacc = st.columns(2)
        with col_acc:
            st.markdown(f"""
                <div style='background:#192132;padding:10px 0;border-radius:10px;text-align:center;'>
                    <span style='font-size:1.1rem;color:#f87171;font-weight:700;'>Accident Probability</span><br>
                    <span style='font-size:1.5rem;color:#f1f5f9;font-weight:800;'>{accident_prob:.1%}</span>
                </div>
            """, unsafe_allow_html=True)
        with col_nonacc:
            st.markdown(f"""
                <div style='background:#192132;padding:10px 0;border-radius:10px;text-align:center;'>
                    <span style='font-size:1.1rem;color:#38bdf8;font-weight:700;'>Non Accident Probability</span><br>
                    <span style='font-size:1.5rem;color:#f1f5f9;font-weight:800;'>{non_accident_prob:.1%}</span>
                </div>
            """, unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:10px; font-size:1.1rem; color:#f1f5f9;'><b>Overall Confidence:</b> {confidence:.1%}</div>", unsafe_allow_html=True)
        st.progress(confidence)
        st.markdown('</div>', unsafe_allow_html=True)
        
        return prediction_label, color_code
    
    @staticmethod
    def render_analysis_summary(
        is_accident: bool,
        confidence: float,
        prediction_label: str,
        color_code: str,
        confidence_threshold: float
    ):
        """Render analysis summary statistics"""
        st.divider()
        st.subheader("📋 Analysis Summary")
        # Only show summary if all values are present and not None
        if (
            confidence is not None and prediction_label and
            isinstance(confidence, (float, int)) and
            prediction_label not in (None, "")
        ):
            confidence_level = (
                "High" if confidence >= 0.8
                else "Medium" if confidence >= 0.6
                else "Low"
            )
            alert_status = (
                "⚠️ YES"
                if (is_accident and confidence > confidence_threshold)
                else "✓ NO"
            )
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                st.markdown(f"**Prediction**<br><span style='font-size:1.1rem'>{color_code} {prediction_label}</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                st.markdown(f"**Confidence**<br><span style='font-size:1.1rem'>{confidence:.1%}</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                st.markdown(f"**Confidence Level**<br><span style='font-size:1.1rem'>{confidence_level}</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                st.markdown(f"**Alert Status**<br><span style='font-size:1.1rem'>{alert_status}</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            # Alert message
            if is_accident and confidence > confidence_threshold:
                st.error(f"⚠️ ALERT: Accident detected with {confidence:.1%} confidence!")
            elif is_accident and confidence <= confidence_threshold:
                st.warning(
                    f"⚠️ Borderline: Accident probability {confidence:.1%} "
                    f"below threshold ({confidence_threshold:.0%})"
                )
        else:
            st.info("No analysis summary available.")
    
    @staticmethod
    def render_footer():
        """Render application footer"""
        st.markdown("""
        <div style='width:100%;text-align:center;margin-top:2rem;margin-bottom:0.5rem;'>
            <span style='margin-right:1.5rem;'>⏰ {}</span>
            <span style='margin-right:1.5rem;'>🚀 Powered by TensorFlow & Streamlit</span>
            <span>🔒 Predictions with confidence scores</span>
        </div>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

# ============================================================================
# IMAGE ANALYSIS MODE
# ============================================================================

def run_image_analysis_mode(
    model_manager: ModelManager,
    prediction_engine: PredictionEngine,
    confidence_threshold: float
):
    """Run image analysis mode"""
    st.header("🖼️ Image Upload & Analysis")
    
    with st.container():
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
            <span style='font-size:2rem;'>📤</span>
            <span style='font-weight:700; font-size:1.2rem;'>Upload CCTV Image</span>
        </div>
        <div style='color:#38bdf8; font-size:1rem; margin-bottom:0.5rem;'>
            Supported formats: <b>JPG, PNG, BMP</b> &nbsp;|&nbsp; Max size: <b>500MB</b>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "",
            type=list(Config.ALLOWED_IMAGE_TYPES),
            help="Choose a traffic/CCTV image for analysis"
        )
    
    if uploaded_file is not None:
        try:
            # Validate file size
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > Config.MAX_FILE_SIZE_MB:
                st.error(f"❌ File too large. Max size: {Config.MAX_FILE_SIZE_MB}MB")
                return
            
            # Load image
            image_pil = Image.open(uploaded_file).convert('RGB')
            
            # Validate image
            if not ImageProcessor.validate_image(image_pil):
                st.error("❌ Invalid image format")
                return
            
            # Create columns
            col_image, col_results = st.columns([1.2, 1])
            
            with col_image:
                st.subheader("📸 Uploaded Image")
                st.image(image_pil, use_container_width=True, caption="Original CCTV Frame")
            
            with col_results:
                st.subheader("🎯 Prediction Results")
                
                # Make prediction
                with st.spinner("🔍 Analyzing..."):
                    is_accident, confidence, accident_prob, non_accident_prob = \
                        prediction_engine.predict_image(image_pil)
                
                if confidence is not None:
                    prediction_label, color_code = UIComponents.render_prediction_result(
                        is_accident, confidence, accident_prob, non_accident_prob
                    )
                    
                    UIComponents.render_analysis_summary(
                        is_accident,
                        confidence,
                        prediction_label,
                        color_code,
                        confidence_threshold
                    )
        
        except Exception as e:
            logger.error(f"Error in image analysis mode: {str(e)}", exc_info=True)
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# VIDEO ANALYSIS MODE
# ============================================================================

def run_video_analysis_mode(
    model_manager: ModelManager,
    prediction_engine: PredictionEngine,
    video_processor: VideoProcessor,
    report_generator: ReportGenerator,
    confidence_threshold: float
):
    """Run video analysis mode"""
    st.header("📹 Video Upload & Frame-by-Frame Analysis")
    
    uploaded_video = st.file_uploader(
        "Upload Video File",
        type=list(Config.ALLOWED_VIDEO_TYPES),
        help="Choose a video file for accident detection"
    )
    
    if uploaded_video is not None:
        try:
            # Validate file size
            file_size_mb = len(uploaded_video.getvalue()) / (1024 * 1024)
            if file_size_mb > Config.MAX_FILE_SIZE_MB:
                st.error(f"❌ File too large. Max size: {Config.MAX_FILE_SIZE_MB}MB")
                return
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            st.info("🔍 Processing video... This may take a few minutes.")
            
            # Process video
            video_results = video_processor.process(video_path, confidence_threshold)
            
            if video_results is None:
                return
            
            st.success("✅ Video processing complete!")
            
            # Display results
            _display_video_results(video_results, confidence_threshold, report_generator)
        
        except Exception as e:
            logger.error(f"Error in video analysis mode: {str(e)}", exc_info=True)
            st.error(f"❌ Error: {str(e)}")
        
        finally:
            # Cleanup
            if 'video_path' in locals() and os.path.exists(video_path):
                os.remove(video_path)
                logger.debug("Temporary video file cleaned up")

def _display_video_results(
    video_results: Dict[str, Any],
    confidence_threshold: float,
    report_generator: ReportGenerator
):
    """Display video analysis results"""
    st.divider()
    st.subheader("📊 Video Analysis Results")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Frames", video_results['total_frames'])
    
    with col2:
        duration_mins = video_results['duration'] / 60
        st.metric("Duration", f"{duration_mins:.1f} min")
    
    with col3:
        st.metric("FPS", f"{video_results['fps']:.0f}")
    
    with col4:
        st.metric("🚨 Accidents Detected", len(video_results['accident_frames']))
    
    st.divider()
    
    # Accident timeline
    if video_results['accident_frames']:
        st.subheader("🚨 Accident Timeline")
        
        accident_data = []
        for idx, acc in enumerate(video_results['accident_frames'], 1):
            minutes = int(acc['timestamp'] // 60)
            seconds = int(acc['timestamp'] % 60)
            accident_data.append({
                "Incident #": idx,
                "Timestamp": f"{minutes:02d}:{seconds:02d}",
                "Frame #": acc['frame_number'],
                "Confidence": f"{acc['confidence']:.1%}"
            })
        
        st.dataframe(accident_data, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No accidents detected in video")
    
    st.divider()
    
    # Statistics
    st.subheader("📈 Detection Statistics")
    
    all_results = video_results['all_results']
    total_analyzed = len(all_results)
    accident_predictions = sum(1 for r in all_results if r['is_accident'])
    safe_predictions = total_analyzed - accident_predictions
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.markdown(f"**Frames Analyzed**\n\n{total_analyzed}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        accident_pct = (accident_predictions / total_analyzed * 100) if total_analyzed > 0 else 0
        st.markdown(f"**Accident Frames**\n\n{accident_predictions} ({accident_pct:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        safe_pct = (safe_predictions / total_analyzed * 100) if total_analyzed > 0 else 0
        st.markdown(f"**Safe Frames**\n\n{safe_predictions} ({safe_pct:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if video_results['accident_frames']:
        avg_confidence = np.mean([a['confidence'] for a in video_results['accident_frames']])
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.markdown(f"**Avg Accident Confidence**\n\n{avg_confidence:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Report generation
    st.divider()
    st.subheader("💾 Export Results")
    
    if st.button("📥 Generate Summary Report", use_container_width=True):
        report = report_generator.generate_text_report(video_results, confidence_threshold)
        
        st.text_area("Report Preview", report, height=300)
        
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"accident_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    try:
        # Page configuration
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout=Config.LAYOUT,
            initial_sidebar_state="expanded"
        )
        
        # Apply styling
        apply_custom_css()
        
        # Render header
        UIComponents.render_header()
        
        # Load model
        model_manager = get_model_manager()
        
        if not model_manager.is_loaded():
            st.error("❌ Model failed to load. Please check the model file path.")
            st.stop()
        
        # Render sidebar
        mode, confidence_threshold = UIComponents.render_sidebar_settings()
        
        # Initialize components
        prediction_engine = PredictionEngine(model_manager)
        video_processor = VideoProcessor(model_manager, prediction_engine)
        report_generator = ReportGenerator()
        
        # Run selected mode
        if mode == "🖼️ Image Analysis":
            run_image_analysis_mode(
                model_manager,
                prediction_engine,
                confidence_threshold
            )
        else:
            run_video_analysis_mode(
                model_manager,
                prediction_engine,
                video_processor,
                report_generator,
                confidence_threshold
            )
        
        # Render footer
        UIComponents.render_footer()
        
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}", exc_info=True)
        st.error(f"❌ Critical error: {str(e)}")

if __name__ == "__main__":
    main()