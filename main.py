import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn

device = ("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        vector = self.flatten(x)
        logits = self.linear_relu_stack(vector)
        return logits


# st.title('_Image Caption Generator_')
html_temp = """
    <div style="background:#522B72 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Classify FashionMNIST image</h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
image = st.file_uploader("Upload an image",
                         type=['png', 'jpg'],
                         help="You can generate caption for only one image at a time")
if image is not None:
    image = Image.open(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
    image = transform(image).unsqueeze(0)
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(r"C:\Users\Manan Mehta\FashionMNISTmodel.pth", map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        pred = model(image)
        predicted = classes[pred[0].argmax(0)]
if st.button("Classify image"):
    st.text(predicted)
