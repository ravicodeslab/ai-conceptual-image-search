# 🧠🔍 AI Powered Conceptual Image Search

A deep learning based visual search engine that understands *conceptual similarity* between images using **ResNet50 embeddings** and **cosine similarity (vector arithmetic)**.

Instead of comparing pixels, this system compares high-dimensional feature vectors to find visually similar images.

---

## 🚀 What This Project Does

This application allows users to:

🔎 **Search in Dataset**  
Upload an image → Find the top visually similar images from a dataset.

🆚 **Compare Two Images**  
Upload two images → Get a similarity score between them.

The system captures semantic features like:
- Shape
- Texture
- Style
- Object structure

Not just color matching.

---

## 🧠 How It Works (Under the Hood)

 Image → Preprocessed to 224×224  
 Passed through pretrained **ResNet50 (ImageNet)**  
 Extract 2048-dimensional feature vector  
 Normalize vectors  
 Compute cosine similarity:

\[
Similarity = (A · B) / (||A|| ||B||)
\]

Higher score → More conceptually similar.

---

## 🏗️ Tech Stack

- 🐍 Python  
- 🔥 PyTorch  
- 🌐 Streamlit  
- 📦 NumPy  
- 🖼️ Pillow  

---

## 📂 Project Structure


---

## ⚡ Features

✔ Pretrained ResNet50 feature extraction  
✔ Manual cosine similarity (no sklearn shortcut)  
✔ Dataset indexing with NumPy caching  
✔ Streamlit interactive UI  
✔ Two operational modes  
✔ Scalable structure  

---

## ▶️ How To Run

```bash
pip install -r requirements.txt
streamlit run app.py
