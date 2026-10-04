import io
from PIL import Image
import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="MNIST Digit Recognition", page_icon="🔢", layout="centered"
)

st.title("🔢 MNIST Digit Recognition System")
st.write(
    "Draw on the canvas or **drag & drop** an image to predict using our **FastAPI + ANN Model**."
)

FASTAPI_URL = "http://127.0.0.1:8000/predict"

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

          files = {
              "file": (
                  "canvas.png",
                  img_byte_arr.getvalue(),
                  "image/png",
              )
          }

          # Timeout hata diya hai taake prediction complete ho sake
          response = requests.post(FASTAPI_URL, files=files)

          if response.status_code == 200:
            result = response.json()
            st.success("Analysis Complete!")
            st.metric(
                label="Predicted Digit", value=str(result["predicted_digit"])
            )
            st.metric(
                label="Confidence Level", value=f"{result['confidence']}%"
            )
          else:
            st.error(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
          st.error(f"Connection Error: {e}")
    else:
      st.warning("Pehle Canvas par koi digit draw karein!")

# ------------------------------------------------------------------------------
# TAB 2: DRAG & DROP / UPLOAD
# ------------------------------------------------------------------------------
with tab2:
  st.subheader("Drag & Drop Image Here")

  uploaded_file = st.file_uploader(
      "Drag and drop file here (PNG, JPG, JPEG)",
      type=["png", "jpg", "jpeg"],
      accept_multiple_files=False,
  )

  if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
      st.image(
          uploaded_file, caption="Uploaded Image", use_column_width=True
      )

    with col2:
      if st.button("Predict Uploaded Image 🚀", key="btn_upload"):
        with st.spinner("Analyzing File..."):
          try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            # Timeout hata diya hai taake response aaram se receive ho sake
            response = requests.post(FASTAPI_URL, files=files)

            if response.status_code == 200:
              result = response.json()
              st.success("Analysis Complete!")
              st.metric(
                  label="Predicted Digit", value=str(result["predicted_digit"])
              )
              st.metric(
                  label="Confidence Level", value=f"{result['confidence']}%"
              )
            else:
              st.error(f"API Error ({response.status_code}): {response.text}")
          except Exception as e:
            st.error(f"Connection Error: {e}")