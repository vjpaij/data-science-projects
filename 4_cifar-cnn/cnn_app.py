import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import torchvision.transforms as transforms
import Net

# Define the Net class (assuming it's a simple CNN, adjust as needed)
# class Net(nn.Module):
#     def __init__(self, dropout_rate=0.5): # Added dropout_rate as a parameter
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(128 * 4 * 4, 512)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(512, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# Load the model (adjust path if needed)
try:
    model = Net()  # Create an instance of the Net class
    model.load_state_dict(torch.load('cifar_net.pth', map_location=torch.device('mps'))) # Load on CPU
    model.eval()  # Set to evaluation mode
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to predict the image class
def predict_image(image):
    img = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return classes[predicted[0]]

# Streamlit app
st.title('CIFAR-10 Image Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make a prediction
    predicted_class = predict_image(image)
    st.write(f'**Prediction:** {predicted_class}')
    st.success(f'The image is classified as: {predicted_class}')
else:
    st.info("Please upload an image to classify.")
