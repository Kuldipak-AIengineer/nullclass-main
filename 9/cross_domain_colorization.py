import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np

# ========== Domain-specific Colorizer Models ==========
class SketchColorizer(nn.Module):
    def __init__(self):
        super(SketchColorizer, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class InfraredColorizer(nn.Module):
    def __init__(self):
        super(InfraredColorizer, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ========== Preprocessing ==========
def preprocess_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((256, 256))
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return tensor, img

# ========== GUI Functionality ==========
class ColorizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cross-Domain Image Colorization")

        self.domain = tk.StringVar()
        self.image_path = None

        ttk.Label(root, text="Select Domain:").grid(row=0, column=0, padx=5, pady=5)
        domain_menu = ttk.Combobox(root, textvariable=self.domain, values=["Sketch", "Infrared"])
        domain_menu.grid(row=0, column=1, padx=5, pady=5)
        domain_menu.current(0)

        ttk.Button(root, text="Choose Image", command=self.choose_image).grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Button(root, text="Colorize", command=self.colorize_image).grid(row=2, column=0, columnspan=2, pady=5)

    def choose_image(self):
        self.image_path = filedialog.askopenfilename()

    def colorize_image(self):
        if not self.image_path:
            print("No image selected!")
            return
        img_tensor, grayscale_img = preprocess_image(self.image_path)

        domain = self.domain.get()
        if domain == "Sketch":
            model = SketchColorizer()
        elif domain == "Infrared":
            model = InfraredColorizer()
        else:
            print("Unknown domain")
            return

        model.eval()
        with torch.no_grad():
            colorized = model(img_tensor)
        
        self.display_images(grayscale_img, colorized.squeeze().permute(1, 2, 0).numpy())

    def display_images(self, gray_img, color_img):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(gray_img, cmap="gray")
        plt.title("Input (Grayscale)")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(color_img)
        plt.title("Output (Colorized)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# ========== Run the App ==========
if __name__ == "__main__":
    root = tk.Tk()
    app = ColorizationApp(root)
    root.mainloop()
