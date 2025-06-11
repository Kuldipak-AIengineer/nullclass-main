# nullclass-main

# 🎨 NullClass Internship Projects – Image Colorization & Computer Vision

This repository contains the complete set of tasks completed during my internship at **NullClass EdTech**, focusing on advanced **image colorization**, **computer vision**, and **interactive GUI applications** using Python.

## 👨‍💻 Author
**Kuldipak Vasudev Patil**  
Email: [kuldipak2228@gmail.com](mailto:kuldipak2228@gmail.com)  
GitHub: [@Kuldipak-AIengineer](https://github.com/Kuldipak-AIengineer)

---

## 📁 Project Structure (Tasks 1–9)

| Task | Title | Description |
|------|-------|-------------|
| **1** | Visualizing the Colorization Process | Visualizes intermediate stages of colorization to understand how grayscale images are progressively turned into colored outputs. |
| **2** | Colorize Different Image Categories | Applies the model to a variety of image types like portraits, animals, and landscapes. |
| **3** | Conditional Image Colorization | Enables user-defined colors (e.g., blue sky, green grass) using a GUI for interactive control. |
| **4** | Dataset Augmentation for Colorization | Uses augmentation (rotation, flip, brightness) to improve model performance. Includes before-and-after comparison. |
| **5** | Artistic Style Transfer in Colorization | Blends style transfer with colorization, allowing users to choose artistic styles via GUI. |
| **6** | Interactive User-Guided Colorization | Users can select image regions and specify colors dynamically through an interactive GUI. |
| **7** | Real-time Multi-Object Colorization | Employs semantic segmentation to colorize multiple real-time objects like trees, cars, etc., in video streams. |
| **8** | Time-Based Historical Colorization | Detects the historical era (e.g., 1900s, 1950s) and colorizes images using era-specific palettes. |
| **9** | Cross-Domain Image Colorization | Colorizes sketches, infrared images, and other domain-specific inputs using appropriate techniques. |

---

## 🧰 Tech Stack

- **Python 3.x**
- **OpenCV**, **MediaPipe**, **Dlib**
- **PyTorch**, **YOLOv8**, **Ultralytics**
- **GUI Libraries:** Tkinter / PyQt5
- **Deep Learning & Colorization Techniques**
- **Semantic Segmentation**, **Style Transfer**

---

## ⚠️ Note About Files

Due to GitHub's file size limit, some files were removed from version control:
- Pretrained models (`*.pt`)
- Landmark predictors (`*.dat`)
- Large video files (`*.mp4`)

> You can manually download these files from [Google Drive](#) *(link to be added)* and place them in the appropriate folders.

---

## 🛠️ Getting Started

```bash
# Clone this repository
git clone https://github.com/Kuldipak-AIengineer/nullclass-main.git
cd nullclass-main

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
