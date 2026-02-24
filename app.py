import os
import streamlit as st
from PIL import Image
from dataset import ImageDataset
from similarity import cosine_similarity

st.set_page_config(page_title="AI Conceptual Image Search", layout="wide")

st.title("AI Powered Conceptual Image Search")
st.write("Choose a mode below to search similar images or compare two images.")

# ---------- Load Dataset (Cached) ----------
@st.cache_resource
def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset_images")
    return ImageDataset(dataset_path)

dataset = load_dataset()

# ---------- Mode Selection ----------
mode = st.radio(
    "Select Mode",
    ["Search Similar Results", "Compare Two Images"]
)

# ============================================================
# MODE 1: SEARCH DATASET
# ============================================================
if mode == "Search Similar Results":

    uploaded_file = st.file_uploader(
        "Upload an image to search",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        query_image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Query Image")
        st.image(query_image, width=300)

        with st.spinner("Searching for similar images..."):
            results = dataset.search(query_image, top_k=3)

        st.subheader("Top Matches")
        cols = st.columns(3)

        for col, (score, path) in zip(cols, results):
            with col:
                st.image(path, use_container_width=True)
                st.markdown(f"**Similarity:** {score:.4f}")

# ============================================================
# MODE 2: COMPARE TWO IMAGES
# ============================================================
elif mode == "Compare Two Images":

    col1, col2 = st.columns(2)

    with col1:
        file1 = st.file_uploader(
            "Upload Image 1",
            type=["jpg", "jpeg", "png"],
            key="img1"
        )

    with col2:
        file2 = st.file_uploader(
            "Upload Image 2",
            type=["jpg", "jpeg", "png"],
            key="img2"
        )

    if file1 and file2:
        img1 = Image.open(file1).convert("RGB")
        img2 = Image.open(file2).convert("RGB")

        st.subheader("Uploaded Images")
        st.image([img1, img2], width=250)

        vec1 = dataset.extractor.extract(img1)
        vec2 = dataset.extractor.extract(img2)

        score = cosine_similarity(vec1, vec2)

        st.subheader(f"Similarity Score: {score:.4f}")