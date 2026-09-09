import streamlit as st 
import cv2 
import numpy as np 
import pickle
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# SIMPLE CSS
# =========================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-top: 20px;
    }

    .result h1 {
        font-size: 38px;
        margin: 10px 0;
    }

    .result p {
        font-size: 16px;
        margin: 5px;
    }

    .footer {
        text-align: center;
        color: #777;
        padding: 30px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🚗 Vehicle Image Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a vehicle image and let Machine Learning identify it.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL SETTINGS
# =========================================================

MODEL_PATH = "vehicle_decision_tree_model.pkl"

IMG_SIZE = 64


# =========================================================
# VEHICLE CLASSES
# =========================================================

class_names = {
    0: "Auto Rickshaws",
    1: "Bikes",
    2: "Cars",
    3: "Motorcycles",
    4: "Planes",
    5: "Ships",
    6: "Trains"
}


# =========================================================
# EMOJIS
# =========================================================

vehicle_emojis = {
    "Auto Rickshaws": "🛺",
    "Bikes": "🚲",
    "Cars": "🚘",
    "Motorcycles": "🏍️",
    "Planes": "✈️",
    "Ships": "🚢",
    "Trains": "🚆"
}


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    clf = load_model()

except FileNotFoundError:

    st.error(
        "❌ vehicle_decision_tree_model.pkl file nahi mili. "
        "Model file ko app.py ke same folder mein rakhein."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Model load nahi ho saka: {e}"
    )

    st.stop()


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    img = np.array(image)

    # Grayscale image
    if len(img.shape) == 2:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )

    # RGBA image
    elif img.shape[2] == 4:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2RGB
        )

    # RGB to BGR
    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    # Resize
    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    # Flatten
    img = img.flatten()

    # Model input
    img = img.reshape(1, -1)

    # Prediction
    prediction = clf.predict(img)

    predicted_class = int(prediction[0])

    return class_names.get(
        predicted_class,
        "Unknown"
    )


# =========================================================
# MAIN COLUMNS
# =========================================================

left, right = st.columns(2)


# =========================================================
# LEFT COLUMN
# =========================================================

with left:

    st.subheader("📤 Upload Vehicle Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    image = None

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            )

            # Always convert to RGB
            if image.mode != "RGB":

                image = image.convert("RGB")

            st.image(
                image,
                caption="Uploaded Vehicle Image"
            )

            st.success(
                "✅ Image uploaded successfully!"
            )

        except Exception as e:

            st.error(
                f"❌ Image open nahi ho saki: {e}"
            )


# =========================================================
# RIGHT COLUMN
# =========================================================

with right:

    st.subheader("🤖 Classification")

    if image is not None:

        if st.button("🚀 Classify Vehicle"):

            try:

                with st.spinner(
                    "🔍 Image analyze ho rahi hai..."
                ):

                    result = predict_image(
                        image
                    )

                emoji = vehicle_emojis.get(
                    result,
                    "🚗"
                )

                # =================================================
                # RESULT
                # =================================================

                st.success("Prediction completed!")

                st.markdown(
                    f"""
                    <div class="result">
                        <p>🤖 AI CLASSIFICATION RESULT</p>
                        <h1>{emoji} {result}</h1>
                        <p>Vehicle successfully identified</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"❌ Prediction error: {e}"
                )

    else:

        st.info(
            "👆 Pehle vehicle image upload karein."
        )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader("📊 Model Information")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Vehicle Classes",
        "7"
    )

with c2:
    st.metric(
        "Image Size",
        "64 × 64"
    )

with c3:
    st.metric(
        "Algorithm",
        "Decision Tree"
    )

with c4:
    st.metric(
        "Task",
        "Image Classification"
    )


# =========================================================
# SUPPORTED VEHICLES
# =========================================================

st.divider()

st.subheader("🚘 Supported Vehicle Classes")

v1, v2, v3, v4 = st.columns(4)

with v1:
    st.info("🛺 Auto Rickshaws")

with v2:
    st.info("🚲 Bikes")

with v3:
    st.info("🚘 Cars")

with v4:
    st.info("🏍️ Motorcycles")


v5, v6, v7 = st.columns(3)

with v5:
    st.info("✈️ Planes")

with v6:
    st.info("🚢 Ships")

with v7:
    st.info("🚆 Trains")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🚗 Vehicle Image Classifier | "
    "Powered by Machine Learning & Streamlit | "
    "Decision Tree Image Classification"
)