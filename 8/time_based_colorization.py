import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog, QComboBox, QHBoxLayout, QPixmap
from PyQt5.QtGui import QImage, QPixmap
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ---- Load Era Classification Model ----
class EraClassifier(torch.nn.Module):
    def __init__(self):
        super(EraClassifier, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 56 * 56, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4)
        )
        
    def forward(self, x):
        return self.conv(x)

era_labels = ['1900s', '1930s', '1950s', '1970s']
era_model = EraClassifier()
era_model.load_state_dict(torch.load('era_classifier.pth', map_location='cpu'))
era_model.eval()

# ---- Dummy Colorization Function (Replace with real one) ----
def colorize_image(gray_img, era):
    # Dummy: apply slight tint depending on era
    tint = {
        '1900s': (0.8, 0.7, 0.6),
        '1930s': (0.9, 0.8, 0.7),
        '1950s': (1.0, 0.9, 0.8),
        '1970s': (1.1, 1.0, 0.9),
    }
    gray_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR) / 255.0
    r, g, b = tint[era]
    colorized = np.clip(gray_3ch * [b, g, r], 0, 1)
    return (colorized * 255).astype(np.uint8)

# ---- Era Detection ----
def detect_era(gray_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    pil_img = Image.fromarray(gray_img)
    img_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        outputs = era_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        era = era_labels[predicted.item()]
    return era

# ---- PyQt5 GUI ----
class ColorizationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Historical Photo Colorization")
        self.setGeometry(100, 100, 1200, 600)

        self.image_label = QLabel("Upload a grayscale image...")
        self.combo = QComboBox()
        self.combo.addItem("Auto")
        self.combo.addItems(era_labels)

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.load_image)
        self.combo.currentIndexChanged.connect(self.update_display)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.combo)
        layout.addWidget(self.upload_btn)
        self.setLayout(layout)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            self.gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.auto_era = detect_era(self.gray_img)
            self.colorized_auto = colorize_image(self.gray_img, self.auto_era)
            self.display_images()

    def update_display(self):
        if hasattr(self, 'gray_img'):
            selected_era = self.combo.currentText()
            if selected_era == "Auto":
                self.display_images()
            else:
                self.manual_colorized = colorize_image(self.gray_img, selected_era)
                self.display_images(manual_override=True)

    def display_images(self, manual_override=False):
        img1 = cv2.cvtColor(self.gray_img, cv2.COLOR_GRAY2BGR)
        img2 = self.colorized_auto
        img3 = colorize_image(self.gray_img, self.combo.currentText()) if manual_override else img2

        # Combine images side by side
        combined = np.hstack((img1, img2, img3))
        qimg = QImage(combined.data, combined.shape[1], combined.shape[0], combined.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaledToWidth(1000))

# ---- Run the App ----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ColorizationApp()
    window.show()
    sys.exit(app.exec_())
