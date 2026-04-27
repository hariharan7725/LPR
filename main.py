import streamlit as st
import cv2
import numpy as np
import torch
import easyocr
from collections import defaultdict
from ultralytics import YOLO


# =========================================================
# MODULE 1: PAGE SETUP + UI STYLING
# ---------------------------------------------------------
# This module handles:
# 1. Streamlit page configuration
# 2. Custom CSS styling for titles, result card, and button
# =========================================================

# Set browser tab title, icon, and wide layout
st.set_page_config(
    page_title="License Plate Detection",
    page_icon="🔢",
    layout="wide"
)

# Apply custom styling to improve the page appearance
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    padding-top: 1.5rem;
    margin-bottom: 0.2rem;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 1.5rem;
}
.result-box {
    padding: 22px;
    border-radius: 16px;
    background: #111827;
    border: 1px solid #374151;
    margin-top: 10px;
}
.result-label {
    font-size: 18px;
    color: #cbd5e1;
    font-weight: 600;
}
.result-plate {
    font-size: 36px;
    color: #22c55e;
    font-weight: 800;
    margin-top: 8px;
    letter-spacing: 1px;
}
.result-info {
    font-size: 16px;
    color: #d1d5db;
    margin-top: 12px;
}
.conf-badge {
    display: inline-block;
    margin-top: 8px;
    padding: 8px 14px;
    border-radius: 999px;
    background: #f59e0b;
    color: #111827;
    font-weight: 800;
    font-size: 16px;
}
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3rem;
    font-size: 18px;
    font-weight: 700;
}
small {
    visibility: hidden;
}
[data-testid="stFileUploaderDropzoneInstructions"] small {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODULE 2: CONFIGURATION + DEVICE CHECK
# ---------------------------------------------------------
# This module handles:
# 1. Model path configuration
# 2. Device selection (GPU or CPU)
# 3. Main page heading display
# =========================================================

# Path of YOLO license plate detection model
PLATE_MODEL = "yolov8n-license-plate.pt"

# Check whether CUDA GPU is available
# If available -> use GPU
# Else -> use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Main heading shown on the page
st.markdown('<div class="main-title">🔢 License Plate Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload a vehicle image and extract the license plate number</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODULE 3: MODEL LOADING
# ---------------------------------------------------------
# This module handles:
# 1. Loading YOLO detection model
# 2. Loading EasyOCR model
# 3. Caching models so they are not loaded repeatedly
# =========================================================

@st.cache_resource
def load_models():
    if DEVICE == "cuda":
        # Load YOLO model on GPU for faster detection
        plate_model = YOLO(PLATE_MODEL).to("cuda")
        try:
            # fuse() can improve inference speed in some cases
            plate_model.fuse()
        except Exception:
            pass

        # Load EasyOCR on GPU
        reader = easyocr.Reader(['en'], gpu=True)
    else:
        # Load YOLO model on CPU
        plate_model = YOLO(PLATE_MODEL)

        # Load EasyOCR on CPU
        reader = easyocr.Reader(['en'], gpu=False)

    return plate_model, reader

# Show loading spinner while models are loaded
with st.spinner("⏳ Loading models..."):
    plate_model, reader = load_models()


# =========================================================
# MODULE 4: OCR TEXT PROCESSING
# ---------------------------------------------------------
# This module handles:
# 1. Cleaning OCR text
# 2. Rule-based correction for common OCR mistakes
# =========================================================

def clean_text(text):
    # Keep only letters and digits
    # Remove spaces and special characters
    text = ''.join(filter(str.isalnum, text))
    return text.upper()  # Convert text to uppercase

def correct_plate_format(text):
    # Expected Indian plate pattern positions:
    # letters -> 0,1,4,5
    # digits  -> 2,3,6,7,8,9
    letter_pos = [0, 1, 4, 5]
    digit_pos = [2, 3, 6, 7, 8, 9]

    # Convert commonly misread letters into digits
    letter_to_digit = {
        'O': '0', 'Q': '0', 'D': '0',
        'I': '1', 'L': '1',
        'Z': '2',
        'A': '4',
        'S': '5',
        'G': '6',
        'B': '8'
    }

    # Convert commonly misread digits into letters
    digit_to_letter = {
        '0': 'O',
        '1': 'I',
        '2': 'Z',
        '4': 'A',
        '5': 'S',
        '6': 'G',
        '8': 'B'
    }

    text = list(text)

    # If text length is too short, return as it is
    if len(text) < 10:
        return "".join(text)

    # Correct digit positions
    for i in digit_pos:
        if i < len(text) and text[i] in letter_to_digit:
            text[i] = letter_to_digit[text[i]]

    # Correct letter positions
    for i in letter_pos:
        if i < len(text) and text[i] in digit_to_letter:
            text[i] = digit_to_letter[text[i]]

    return "".join(text)


# =========================================================
# MODULE 5: IMAGE PREPROCESSING
# ---------------------------------------------------------
# This module handles:
# 1. Grayscale conversion
# 2. Noise reduction
# 3. Image enlargement
# 4. Thresholding before OCR
# =========================================================

def preprocess_plate(plate_crop):
    # Convert color image to grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # Reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Resize image to make characters larger for OCR
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Apply Otsu thresholding to separate text from background
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return gray, thresh


# =========================================================
# MODULE 6: MAIN UI + IMAGE PROCESSING + DETECTION OUTPUT
# ---------------------------------------------------------
# This module handles:
# 1. File upload
# 2. Image preview
# 3. ROI creation
# 4. YOLO plate detection
# 5. OCR on detected plate
# 6. Confidence voting
# 7. Final result display
# =========================================================

st.markdown("---")
st.subheader("📥 Upload Vehicle Image")

# Upload input image from user
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded_file is not None:
    # Convert uploaded file bytes into OpenCV image format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Stop if image decoding failed
    if frame is None:
        st.stop()

    # Create two columns:
    # left  -> image preview
    # right -> process button
    left, right = st.columns([2, 1])

    with left:
        # Show uploaded original image
        st.image(frame, caption="📷 Uploaded Vehicle Image", channels="BGR", use_column_width=True)

    with right:
        st.markdown("### 🚘 Ready for Plate Scan")
        process_btn = st.button("🔍 Detect Plate")

    if process_btn:
        with st.spinner("🔎 Detecting plate and extracting text..."):
            # Get image height and width
            H, W = frame.shape[:2]

            # ---------------- ROI CREATION ----------------
            # Restrict detection area to center-lower region
            # This helps reduce false detections and speeds up processing
            expand_ratio = 0.1
            roi_x1 = max(0, int(W * 0.25) - int(W * expand_ratio))
            roi_x2 = min(W, int(W * 0.75) + int(W * expand_ratio))
            roi_y1, roi_y2 = int(H * 0.30), int(H * 0.90)

            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            # Stop if ROI extraction failed
            if roi.size == 0:
                st.stop()

            # ---------------- LICENSE PLATE DETECTION ----------------
            # Run YOLO model on ROI only
            plate_results = plate_model(
                roi,
                conf=0.4,  # minimum confidence for detection
                device=0 if DEVICE == "cuda" else "cpu",
                verbose=False
            )

            # Variables to keep best final result
            detected = False
            best_plate = None
            best_conf = -1
            best_crop = None
            best_gray = None
            best_thresh = None

            # Loop through all YOLO results
            for p in plate_results:
                if p.boxes is None:
                    continue

                # Loop through all detected boxes
                for pb in p.boxes.xyxy:
                    px1, py1, px2, py2 = map(int, pb)

                    # Crop only detected plate region
                    plate_crop = roi[py1:py2, px1:px2]

                    if plate_crop.size == 0:
                        continue

                    # ---------------- PREPROCESSING BEFORE OCR ----------------
                    # Convert to grayscale and threshold image for clearer text
                    gray_crop, processed_crop = preprocess_plate(plate_crop)

                    # ---------------- OCR READING ----------------
                    # OCR is performed on thresholded image
                    results = reader.readtext(processed_crop)
                    ocr_confidences = defaultdict(list)

                    # Process each OCR result
                    for bbox, text, confidence in results:
                        cleaned = clean_text(text)

                        # Ignore too-short OCR text
                        if len(cleaned) >= 5:
                            corrected = correct_plate_format(cleaned)

                            # Store confidence for confidence voting
                            ocr_confidences[corrected].append(confidence)

                    # Skip if no valid OCR result found
                    if not ocr_confidences:
                        continue

                    # ---------------- CONFIDENCE VOTING ----------------
                    # Average confidence for each OCR candidate
                    avg_conf = {
                        plate: sum(confs) / len(confs)
                        for plate, confs in ocr_confidences.items()
                    }

                    # Select OCR text with highest average confidence
                    final_plate = max(avg_conf, key=avg_conf.get)
                    final_conf = avg_conf[final_plate]

                    detected = True

                    # Keep only the best overall detected plate
                    if final_conf > best_conf:
                        best_conf = final_conf
                        best_plate = final_plate
                        best_crop = plate_crop
                        best_gray = gray_crop
                        best_thresh = processed_crop

            # ---------------- RESULT DISPLAY ----------------
                        st.markdown("---")
                        st.subheader("🪪 Plate Processing Output")
                        
                        if detected and best_plate is not None:
                        
                            # ===== ROW 1: Cropped + Grayscale =====
                            col1, col2 = st.columns(2)
                        
                            with col1:
                                if best_crop is not None:
                                    st.image(
                                        best_crop,
                                        caption="🪪 Cropped License Plate",
                                        channels="BGR",
                                        use_column_width=True
                                    )
                        
                            with col2:
                                if best_gray is not None:
                                    st.image(
                                        best_gray,
                                        caption="🌑 Grayscale Image",
                                        use_column_width=True
                                    )
                        
                            # ===== ROW 2: Threshold + Result =====
                            col3, col4 = st.columns(2)
                        
                            with col3:
                                if best_thresh is not None:
                                    st.image(
                                        best_thresh,
                                        caption="⬛ Threshold Image",
                                        use_column_width=True
                                    )
                        
                            with col4:
                                st.markdown(f"""
                                <div class="result-box">
                                    <div class="result-label">🔠 Detected Plate Number</div>
                                    <div class="result-plate">{best_plate}</div>
                                    <div class="result-info">🎯 Confidence Score</div>
                                    <div class="conf-badge">{best_conf:.2f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        else:
                            st.markdown(
                                "<p style='color:#f87171; font-size:18px; font-weight:600;'>No license plate detected.</p>",
                                unsafe_allow_html=True
                            )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"