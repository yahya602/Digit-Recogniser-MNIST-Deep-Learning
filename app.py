import io
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

st.set_page_config(
    page_title="MNIST Digit Recognition", page_icon="🔢", layout="centered"
)


# Model Load Function
@st.cache_resource
def load_mnist_model():
  return tf.keras.models.load_model("mnist_model.keras")


MODEL = load_mnist_model()


# Preprocessing Logic
def preprocess_image(image_bytes: bytes) -> np.ndarray:
  image = Image.open(io.BytesIO(image_bytes)).convert("L")

  # Background Invert Check
  img_array_temp = np.array(image)
  corner_avg = (
      img_array_temp[0, 0]
      + img_array_temp[0, -1]
      + img_array_temp[-1, 0]
      + img_array_temp[-1, -1]
  ) / 4.0

  if corner_avg > 127:
    image = ImageOps.invert(image)

  # Resize to 28x28
  image = image.resize((28, 28), Image.Resampling.BILINEAR)
  img_array = np.array(image, dtype=np.float32) / 255.0

  # Model Input Shape Handle
  if len(MODEL.input_shape) == 2 and MODEL.input_shape[1] == 784:
    return img_array.reshape(1, 784)
  else:
    return np.expand_dims(img_array, axis=0)


# UI Layout
st.title("🔢 MNIST Digit Recognition System")
st.write(
    "Draw on the canvas or **drag & drop** an image to predict using our ANN"
    " Model."
)

tab1, tab2 = st.tabs(["✏️ Draw Digit (Canvas)", "📤 Drag & Drop / Upload Image"])

# ------------------------------------------------------------------------------
# TAB 1: CANVAS
# ------------------------------------------------------------------------------
with tab1:
  st.subheader("Interactive Canvas")
  st.write("Mouse se 0-9 tak digit draw karein:")

  canvas_result = st_canvas(
      fill_color="black",
      stroke_width=18,
      stroke_color="white",
      background_color="black",
      height=280,
      width=280,
      drawing_mode="freedraw",
      key="mnist_canvas",
  )

  if st.button("Predict Canvas Drawing 🚀", key="btn_canvas"):
    if (
        canvas_result.image_data is not None
        and canvas_result.image_data[:, :, :3].any()
    ):
      with st.spinner("Analyzing Drawing..."):
        try:
          img = Image.fromarray(canvas_result.image_data.astype("uint8"))
          img_byte_arr = io.BytesIO()
          img.save(img_byte_arr, format="PNG")

          processed_img = preprocess_image(img_byte_arr.getvalue())
          predictions = MODEL.predict(processed_img)
          predicted_class = int(np.argmax(predictions[0]))
          confidence = float(np.max(predictions[0])) * 100

          st.success("Analysis Complete!")
          st.metric(label="Predicted Digit", value=str(predicted_class))
          st.metric(label="Confidence Level", value=f"{round(confidence, 2)}%")
        except Exception as e:
          st.error(f"Error: {e}")
    else:
      st.warning("Pehle Canvas par koi digit draw karein!")

# ------------------------------------------------------------------------------
# TAB 2: DRAG & DROP / UPLOAD
# ------------------------------------------------------------------------------
with tab2:
  st.subheader("Drag & Drop Image Here")

  uploaded_file = st.file_uploader(
      "Drag and drop file here (PNG, JPG, JPEG,WEBP)",
      type=["png", "jpg", "jpeg","webp"],
      accept_multiple_files=False,
  )

  if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
      st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with col2:
      if st.button("Predict Uploaded Image 🚀", key="btn_upload"):
        with st.spinner("Analyzing File..."):
          try:
            processed_img = preprocess_image(uploaded_file.getvalue())
            predictions = MODEL.predict(processed_img)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0])) * 100

            st.success("Analysis Complete!")
            st.metric(label="Predicted Digit", value=str(predicted_class))
            st.metric(
                label="Confidence Level", value=f"{round(confidence, 2)}%"
            )
          except Exception as e:
            st.error(f"Error: {e}")
