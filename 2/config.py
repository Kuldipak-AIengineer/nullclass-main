# Model configurations
MODEL_CONFIG = {
    'confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'input_size': (640, 640)
}

# Animal classifications
CARNIVORES = [
    'lion', 'tiger', 'bear', 'wolf', 'leopard', 'cheetah'
]

HERBIVORES = [
    'deer', 'elephant', 'giraffe', 'horse', 'cow', 'sheep', 'goat'
]

# GUI configurations
GUI_CONFIG = {
    'window_title': 'Animal Detection System',
    'window_size': (1200, 800),
    'max_display_size': (800, 600)
}

# Colors for visualization
COLORS = {
    'carnivore': (0, 0, 255),    # Red
    'herbivore': (0, 255, 0),    # Green
    'text': (255, 255, 255)      # White
}