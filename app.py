import os
import time
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Konfigurasi Halaman 
st.set_page_config(
    page_title="Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="wide"
)

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
        color: white;
    }
    .metric-container {
        display: flex;
        gap: 15px;
        margin-top: 15px;
    }
    .metric-box {
        flex: 1;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 5px;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)

# Arsitektur Custom CNN (Scratch)
class TomatoCustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(TomatoCustomCNN, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                  
            nn.Linear(4096, 256),             
            nn.BatchNorm1d(256),              
            nn.ReLU(inplace=True),            
            nn.Dropout(0.5),                  
            nn.Linear(256, 128),              
            nn.ReLU(inplace=True),            
            nn.Dropout(0.3),                  
            nn.Linear(128, num_classes)       
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = torch.flatten(x, 1)               
        x = self.classifier(x)
        return x

CLASS_NAMES = [
    'Bacterial spot', 'Early blight', 'Late blight', 'Leaf Mold',
    'Septoria leaf spot', 'Spider mites Two-spotted spider mite',
    'Target Spot', 'Tomato Yellow Leaf Curl Virus',
    'Tomato mosaic virus', 'Healthy'
]
NUM_CLASSES = len(CLASS_NAMES)

# Cache & Model Loading
@st.cache_resource
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scratch Model
    custom_model = TomatoCustomCNN(num_classes=NUM_CLASSES)
    custom_path = os.path.join("save_models", "custom_cnn_tomato.pth")
    if os.path.exists(custom_path):
        custom_model.load_state_dict(torch.load(custom_path, map_location=device))
    custom_model.to(device)
    custom_model.eval()

    # Pretrained Model
    resnet_model = models.resnet18(weights=None)
    in_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES)
    )
    resnet_path = os.path.join("save_models", "resnet18_finetuned_tomato.pth")
    if os.path.exists(resnet_path):
        resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet_model.to(device)
    resnet_model.eval()

    return {
        "ResNet18 (Transfer Learning)": resnet_model,
        "TomatoCustomCNN (Scratch CNN)": custom_model
    }, device

models_dict, device = load_all_models()

# Pipeline Transformasi Sesuai Kebutuhan Input Masing-Masing Model
transform_resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_custom = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sidebar Interaktif
st.sidebar.title("Kontrol Model")
selected_model_name = st.sidebar.selectbox("Pilih Model Inferensi:", list(models_dict.keys()))
st.sidebar.markdown(f"**Akselerator Aktif:** `{str(device).upper()}`")

# Antarmuka Utama 
st.title("Tomato Leaf Disease Detection Hub")
st.write("Klasifikasi 10 kondisi daun tomat berbasis PyTorch.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Unggah Sampel Daun")
    uploaded_file = st.file_uploader("Format file: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Citra Input", use_container_width=True)

with col2:
    st.subheader("2. Hasil Diagnosa")
    if uploaded_file is not None:
        if st.button("Jalankan Inferensi", use_container_width=True):
            with st.spinner("Menganalisis citra..."):
                if "ResNet" in selected_model_name:
                    input_tensor = transform_resnet(img).unsqueeze(0).to(device)
                else:
                    input_tensor = transform_custom(img).unsqueeze(0).to(device)
                
                # Benchmark Latensi
                start_time = time.time()
                active_model = models_dict[selected_model_name]
                with torch.no_grad():
                    output = active_model(input_tensor)
                    probabilities = torch.softmax(output[0], dim=0)
                latency_ms = (time.time() - start_time) * 1000

                top_prob, top_idx = torch.topk(probabilities, 1)
                pred_label = CLASS_NAMES[top_idx.item()]
                confidence = top_prob.item() * 100

                if pred_label.lower() == "healthy":
                    st.success(f"### Kondisi: **{pred_label.upper()}** 🌿")
                else:
                    st.error(f"### Terdeteksi: **{pred_label}** ⚠️")

                st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
                st.progress(confidence / 100.0)

                # Metrik Benchmark
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-box">
                        <div class="metric-title">Latency</div>
                        <div class="metric-value">{latency_ms:.2f} ms</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">Model Architecture</div>
                        <div class="metric-value">{"ResNet18" if "ResNet" in selected_model_name else "Custom CNN"}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Silakan pilih gambar di panel sebelah kiri untuk memulai pengujian.")