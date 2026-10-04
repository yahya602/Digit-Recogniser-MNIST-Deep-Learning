# 🔢 MNIST Digit Recognition Web App

An interactive Deep Learning application built with **Streamlit** and **TensorFlow** that predicts handwritten digits (0–9) using an Artificial Neural Network (ANN) model trained on the MNIST dataset.

🚀 **Live Demo:** [digit-recogniser-mnist-deep-learning.streamlit.app](https://digit-recogniser-mnist-deep-learning.streamlit.app/)

---

## ✨ Features

* **Interactive Canvas:** Draw any single digit (0–9) directly on the screen using your mouse or touch input.
* **Image Drag & Drop / Upload:** Upload digit images in `PNG`, `JPG`, `JPEG`, or `WEBP` formats.
* **Automatic Inversion & Resizing:** Automatically handles background colors (light/dark) and resizes images to $28 \times 28$ pixels to match the model input.
* **Real-time Prediction:** Instant output displaying the **Predicted Digit** alongside the model's **Confidence Level (%)**.

---

## 🛠️ Tech Stack

* **Frontend & Deployment:** Streamlit, Streamlit Drawable Canvas
* **Machine Learning / Deep Learning:** TensorFlow, Keras
* **Data & Image Processing:** NumPy, Pillow (PIL)
* **Language:** Python 3.11

---

## 📁 Project Structure

```text
├── app.py                  # Main Streamlit application script
├── mnist_model.keras       # Trained ANN Keras model
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
