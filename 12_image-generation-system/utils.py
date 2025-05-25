import matplotlib.pyplot as plt
import torchvision

def show_images(fake_imgs):
    grid = torchvision.utils.make_grid(fake_imgs[:25], nrow=5, normalize=True)
    plt.figure(figsize=(8,8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.show()