import streamlit as st
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Walnut Leaf XAI – Occlusion",
    layout="centered"
)
# ======================================================
# FILE UPLOADER COLOR CUSTOMIZATION
# ======================================================
st.markdown(
    """
    <style>
    /* File uploader container */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #4CAF50;
        background-color: #f3fbef;
        border-radius: 10px;
        padding: 12px;
    }

    /* Browse files button */
    div[data-testid="stFileUploader"] button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 6px 14px !important;
        font-weight: 600 !important;
        border: none !important;
    }

    /* Hover effect */
    div[data-testid="stFileUploader"] button:hover {
        background-color: #388E3C !important;
        color: white !important;
    }

    /* Uploaded filename text */
    div[data-testid="stFileUploader"] small {
        color: #2E7D32;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# STYLING (BACKGROUND + COMPACT LAYOUT)
# ======================================================
st.markdown("""
<style>
.stApp { background-color: #eef7e6; }

.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.8rem;
}

h1 { font-size: 1.6rem; }
h2 { font-size: 1.2rem; }
h3 { font-size: 1.05rem; }

img { max-height: 240px; object-fit: contain; }
canvas { max-height: 250px !important; }
</style>
""", unsafe_allow_html=True)

st.title("🍃 Walnut Leaf Disease Detection  with Occlusion XAI")

# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "model", "walnut_cnn.keras")
IMG_SIZE = 224

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# ======================================================
# IMAGE UPLOADER
# ======================================================
uploaded_file = st.file_uploader(
    "Upload Walnut Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ======================================================
# OCCLUSION FUNCTION (PUBLICATION-READY)
# ======================================================
def occlusion_sensitivity(img_array, model, patch=32, stride=16):
    heatmap = np.zeros((IMG_SIZE, IMG_SIZE))
    mean_pixel = np.mean(img_array)
    base_pred = model.predict(img_array, verbose=0)[0]

    for y in range(0, IMG_SIZE - patch, stride):
        for x in range(0, IMG_SIZE - patch, stride):

            occluded = img_array.copy()
            occluded[:, y:y+patch, x:x+patch, :] = mean_pixel
            pred = model.predict(occluded, verbose=0)[0]

            gain = np.sum(np.maximum(pred - base_pred, 0))
            heatmap[y:y+patch, x:x+patch] += gain

    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap

# ======================================================
# MAIN PIPELINE
# ======================================================
if uploaded_file:

    # -------- Load image --------
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------- Prediction --------
    pred = model.predict(img_array, verbose=0)
    pred_class = np.argmax(pred)
    confidence = pred[0][pred_class]
    CLASS_NAMES = ["Anthracnose", "Healthy"]

    # ======================================================
    # ROW 1 — Original + Prediction
    # ======================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Leaf")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Model Prediction")
        st.write(f"**Class:** {CLASS_NAMES[pred_class]}")
        st.write(f"**Confidence:** {confidence:.4f}")
    
    # ======================================================
    # OCCLUSION MAP (BASE)
    # ======================================================
    occ_map = occlusion_sensitivity(img_array, model)

    # ======================================================
    # ROW 2 — Occlusion Map + Overlay
    # ======================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Occlusion Sensitivity Map")
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(occ_map, cmap="plasma")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")
        st.pyplot(fig)

    with col2:
        st.subheader("Occlusion Overlay")

        occ_norm = (occ_map - occ_map.min()) / (occ_map.max() - occ_map.min() + 1e-8)
        heat = cv2.applyColorMap(np.uint8(255 * occ_norm), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        overlay = (
            (1 - occ_norm[..., None]) * np.array(img) +
            occ_norm[..., None] * heat
        ).astype(np.uint8)

        st.image(overlay, use_container_width=True)

    # ======================================================
    # ROW 3 — CRITICAL REGION (64×64)
    # ======================================================
    y, x = np.unravel_index(np.argmax(occ_map), occ_map.shape)
    PATCH = 64
    y = min(y, IMG_SIZE - PATCH)
    x = min(x, IMG_SIZE - PATCH)

    occluded_img = img_array.copy()
    occluded_img[:, y:y+PATCH, x:x+PATCH, :] = 0
    occluded_vis = (occluded_img[0] * 255).astype(np.uint8)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Critical Region (64×64)")
        st.image(occluded_vis, use_container_width=True)

    with col2:
        st.subheader("Lesion Contribution Score")
        st.success(f"{np.mean(occ_map):.4f}")

    # ======================================================
    # ROW 4 — MULTI-SCALE OCCLUSION (2 PER ROW)
    # ======================================================
    st.subheader("Multi-Scale Occlusion Analysis")

    scales = [(16,8), (32,16), (48,24), (64,32)]

    for i in range(0, len(scales), 2):
        col1, col2 = st.columns(2)

        for col, (p, s) in zip([col1, col2], scales[i:i+2]):
            occ = occlusion_sensitivity(img_array, model, patch=p, stride=s)

            with col:
                fig, ax = plt.subplots(figsize=(4,4))
                im = ax.imshow(occ, cmap="plasma")
                ax.set_title(f"{p}×{p}, stride {s}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046)
                st.pyplot(fig)
