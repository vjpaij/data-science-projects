import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import Generator, Discriminator
import torch.nn as nn
import torchvision

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),           # Convert to tensor (0-1)
    transforms.Normalize((0.5,), (0.5,))  # Scale to (-1, 1)
])


train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform),
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize models
G = Generator().to(device)
D = Discriminator().to(device)

# Loss & Optimizers
criterion = nn.BCELoss()
G_optim = torch.optim.Adam(G.parameters(), lr=0.0002)
D_optim = torch.optim.Adam(D.parameters(), lr=0.0002)

# Training loop
for epoch in range(100):
    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        
        # Train Discriminator
        D_optim.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Real images loss
        real_preds = D(real_imgs)
        D_real_loss = criterion(real_preds, real_labels)
        
        # Fake images loss
        z = torch.randn(batch_size, 100).to(device)
        fake_imgs = G(z)
        fake_preds = D(fake_imgs.detach())
        D_fake_loss = criterion(fake_preds, fake_labels)
        
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optim.step()
        
        # Train Generator
        G_optim.zero_grad()
        fake_preds = D(fake_imgs)
        G_loss = criterion(fake_preds, real_labels)  # Fool D
        G_loss.backward()
        G_optim.step()

torch.save(G.state_dict(), "generator.pth")

