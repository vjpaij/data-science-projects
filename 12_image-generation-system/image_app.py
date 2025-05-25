import streamlit as st
import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_channels * 32 * 32),  # CIFAR-10: 32x32x3
            nn.Tanh()  # Output in (-1, 1)
        )

    def forward(self, z):
        return self.model(z).view(-1, 3, 32, 32)
    
def generate_image(G, device):
    z = torch.randn(1, 100).to(device)
    fake_img = G(z).squeeze()  # (3, 32, 32)
    
    # Fix 1: Clamp to [-1, 1] then rescale to [0, 1]
    fake_img = torch.clamp(fake_img, -1, 1)  # Force values between -1 and 1
    fake_img = (fake_img + 1) / 2  # Rescale to [0, 1]
    
    return fake_img.permute(1, 2, 0).cpu().detach().numpy()

G = Generator().to(device)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()

st.title("Image Generation System")
if st.button("Generate Image"):
    z = torch.randn(1, 100).to(device)
    fake_img = generate_image(G, device)
    st.image(fake_img, caption="Generated Image", use_column_width=True)