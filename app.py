import streamlit as st
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from imagecaption import generate_caption 
import pandas as pd
import os
from datetime import datetime

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Toxicity Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Toxicity Detection (Image or Text)")

# -------------------------
# CSV Storage
# -------------------------
CSV_FILE = "toxicity_history.csv"

def save_prediction_csv(input_type, original_input, generated_caption,prediction, confidence, image=None):
# The row structure that will be saved in the csv file (database)
    new_row = {
        "Type": input_type,
        "Text Input": original_input,
        "Generated Caption": generated_caption,
        "Prediction": prediction,
        "Confidence": confidence,
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image": image
    }
    #creating a new csv or appending if it is already created
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(CSV_FILE, index=False)
# -------------------------
# Load Toxicity Model (importing the distbert trained adapters to use it in the text prediction)
# -------------------------
@st.cache_resource
def load_toxicity_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model_name = "distilbert-base-uncased"
    adapter_path = "saved_model"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer, toxicity_model, device = load_toxicity_model()

# -------------------------
# User Input 
# -------------------------
st.subheader("Choose Input Method")
input_type = st.radio(
    "Select input type:",
    ("Text", "Image")
)

text_input = ""
image_input = None

if input_type == "Text":
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150
    )

elif input_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file).convert("RGB")
        st.image(image_input, caption="Uploaded Image", use_column_width=True)

# -------------------------
# Analyze Button
# -------------------------
if st.button("Analyze"):

    if input_type == "Text":
        if text_input.strip() == "":
            st.warning("Please enter some text.")
            st.stop()
        text_to_classify = text_input
        original_input = text_input
        generated_caption = None

    else:  
        if image_input is None:
            st.warning("Please upload an image.")
            st.stop()
        with st.spinner("Generating caption from image..."):
            text_to_classify = generate_caption(image_input)

        st.subheader("Generated Caption")
        st.write(text_to_classify)

        original_input = None
        generated_caption = text_to_classify

    # -------- Classify --------
    with st.spinner("Analyzing toxicity..."):
        inputs = tokenizer(
            text_to_classify,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=100
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = toxicity_model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)

        confidence = probs[0][prediction].item()
        label_map = {0: "Non-Toxic", 1: "Toxic"}
        result = label_map[prediction]

    # -------- Display Result --------
    st.subheader("Toxicity Result")
    if result == "Toxic":
        st.error(f"{result} 🚨")
    else:
        st.success(f"{result} ✅")
    st.write(f"Confidence: {confidence:.2%}")

    # -------- Save to CSV --------
    save_prediction_csv(
        input_type=input_type,
        original_input=text_input if input_type == "Text" else None,
        generated_caption=text_to_classify if input_type == "Image" else None,
        prediction=result,
        confidence=confidence,
        image=image_input if input_type == "Image" else None
    )

# -------------------------
# Show History
# -------------------------
st.markdown("---")
st.subheader("📊 Prediction History")

if st.button("Show History"):

    if not os.path.exists(CSV_FILE):
        st.info("No predictions yet.")
    else:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            st.info("No predictions yet.")
        else:
            df["Confidence"] = df["Confidence"].apply(lambda x: f"{float(x):.2%}")
            st.dataframe(
                df.drop(columns=["Image"]),  
                use_container_width=True
            )
