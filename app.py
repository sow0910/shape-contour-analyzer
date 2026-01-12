import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Shape & Contour Analyzer",
    page_icon="🔍",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }

    /* Title */
    h1 {
        color: #f9d342;
        text-align: center;
        font-weight: 700;
    }

    /* Subheaders */
    h2, h3 {
        color: #f1f1f1;
    }

    /* Text */
    p, label {
        color: #e0e0e0;
        font-size: 16px;
    }

    /* File uploader box */
    section[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.08);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Dataframe */
    .stDataFrame {
        background-color: rgba(0, 0, 0, 0.4);
        border-radius: 10px;
    }

    /* Success box */
    .stAlert {
        background-color: rgba(0, 128, 0, 0.25) !important;
        border-left: 5px solid #2ecc71;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HEADER ----------------
st.title("🔍 Shape & Contour Analyzer")
st.write(
    "Upload an image to detect shapes, count objects, and calculate area & perimeter."
)

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"]
)

# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Convert to grayscale safely
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Blur and edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    results = []
    object_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500:
            object_count += 1
            perimeter = cv2.arcLength(cnt, True)

            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            sides = len(approx)

            if sides == 3:
                shape = "Triangle"
            elif sides == 4:
                shape = "Rectangle"
            elif sides > 4:
                shape = "Circle"
            else:
                shape = "Unknown"

            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(
                img,
                shape,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 215, 0),
                2
            )

            results.append(
                [object_count, shape, round(area, 2), round(perimeter, 2)]
            )

    # ---------------- OUTPUT ----------------
    st.subheader("🖼️ Detected Shapes")
    st.image(img, width=900)


    st.subheader("📊 Shape Details")
    df = pd.DataFrame(
        results, columns=["Object No", "Shape", "Area", "Perimeter"]
    )
    st.dataframe(df, use_container_width=True)

    st.success(f"Total Objects Detected: {object_count}")


