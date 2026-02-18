# 🛡️ Toxicity Detection System (Text & Image)

A Streamlit-based application that detects whether input text or image uploaded are **Toxic** or **Non-Toxic** using a fine-tuned DistilBERT model with LoRA and BLIP for image captioning.

---

## 📌 Project Overview

This system allows users to:

* Enter **text** for toxicity classification
* Upload an **image**, generate a caption using BLIP
* Classify the generated caption as **Toxic / Non-Toxic**
* Store prediction history in a CSV file
* Display prediction history inside the Streamlit interface

---

## 🧠 Models Used

* **Toxicity Classifier:** DistilBERT + LoRA (PEFT)
* **Image Captioning Model:** BLIP (Salesforce BLIP Image Captioning Base)
* **Framework:** PyTorch
* **Frontend:** Streamlit

---

## ⚙️ Installation

Install dependencies:

```bash
pip install torch transformers peft streamlit pillow pandas scikit-learn tqdm
```

## 🚀 Run the Application

```bash
streamlit run app.py
```

---

## 🖼️ Application Features

### 1️⃣ Text Toxicity Detection

* User enters text
* Model predicts Toxic or Non-Toxic
* Confidence score is displayed

### 2️⃣ Image Toxicity Detection

* User uploads image
* BLIP generates caption
* Caption is classified
* Result + confidence displayed

### 3️⃣ Prediction History

* All inputs and outputs saved in `toxicity_history.csv`
* History displayed as a table inside Streamlit

---


## 📊 Prediction History

| Type  | Text Input      | Generated Caption                          | Prediction | Confidence | Time                |
|-------|----------------|--------------------------------------------|------------|------------|---------------------|
| Image | —              | a fire is seen from the side of a building| Toxic      | 72.46%     | 2026-02-17 16:33:22 |
| Text  | i will kill you| —                                          | Toxic      | 88.55%     | 2026-02-17 16:33:51 |
| Image | —              | a fire is seen from the side of a building| Toxic      | 72.46%     | 2026-02-18 15:19:31 |
| Text  | murder         | —                                          | Toxic      | 91.20%     | 2026-02-18 15:20:08 |
| Text  | hate           | —                                          | Non-Toxic  | 55.48%     | 2026-02-18 15:32:50 |


---

## 📁 Project Structure

```
├── app.py
├── imagecaption.py
├── toxicity_history.csv
├── saved_model/
├── text.py
├── Toxic_data_cleaned (2)
└── README.md
```

---

## 🗄️ Data Storage

Prediction results are stored in:

```
toxicity_history.csv
```

Stored fields:

* Input Type
* Original Text
* Generated Caption
* Prediction
* Confidence
* Timestamp
* Encoded Image (Base64)

---

## 📌 Technologies Used

* Python 3.x
* PyTorch
* HuggingFace Transformers
* PEFT (LoRA)
* Streamlit
* Pandas
* BLIP (Image Captioning)

---

## 📷 Screenshots

![Home Page](1.png)

![Text Classification](2.png)

![Image Upload](3.png)

![Generated Caption](4.png)

![Prediction History](5.png)





