import streamlit as st
from torchvision import models, transforms
from PIL import Image
import torch

resnet = models.resnet101(pretrained=True)
resnet.eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

st.title('Image Recognition with ResNet-101')

uploaded_file = st.file_uploader("Upload a image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    out = resnet(batch_t)
    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    label = labels[index[0]]
    confidence = percentage[index[0]].item()

    st.write(f"Class: {label}")
    st.write(f"Percentage: {confidence:.2f}%")