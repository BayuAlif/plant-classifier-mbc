import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Tomato Disease Classifier v1.0", layout="centered")
st.title("Tomato Leaf Disease Classifier (v1.0 Baseline)")
st.write("Aplikasi deteksi penyakit daun tomat menggunakan model Deep Learning.")

CLASS_NAMES = [
    'Bacterial spot', 'Early blight', 'Late blight', 'Leaf Mold',
    'Septoria leaf spot', 'Spider mites Two-spotted spider mite',
    'Target Spot', 'Tomato Yellow Leaf Curl Virus',
    'Tomato mosaic virus', 'Healthy'
]

@st.cache_resource
def load_baseline_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(CLASS_NAMES))
    )
    model_path = os.path.join("save_models", "resnet18_finetuned_tomato.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_baseline_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Unggah Sampel Daun Tomat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar Input", width=350)
    
    if st.button("Jalankan Inferensi"):
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs[0], dim=0)
            top_prob, top_idx = torch.topk(probs, 1)
        
        label = CLASS_NAMES[top_idx.item()]
        score = top_prob.item() * 100
        st.write(f"### Hasil Prediksi: **{label}** ({score:.2f}%)")