import sys
import requests
import random
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLineEdit, QPushButton, QLabel, QTabWidget, QTextEdit, 
                            QScrollArea, QFrame, QGridLayout, QMessageBox, QGroupBox,
                            QFormLayout, QCheckBox, QComboBox, QSlider, QFileDialog,
                            QSizePolicy, QSpacerItem, QColorDialog)
from PyQt5.QtCore import Qt, QUrl, QSize
from PyQt5.QtGui import QFont, QDesktopServices, QPixmap, QImage, QColor, QImageReader
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
import cv2
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from PyQt5.QtCore import QThread, pyqtSignal

class ImageEditorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.original_image = None
        self.filename = None
        self.setup_ui()
        
    def setup_ui(self):
        self.layout = QHBoxLayout(self)
        
        # Left panel - Image display
        self.image_frame = QFrame()
        self.image_frame.setFrameShape(QFrame.Box)
        self.image_frame.setStyleSheet("background-color: #f0f0f0;")
        self.image_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #e0e0e0;")
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_label)
        
        self.image_frame_layout = QVBoxLayout(self.image_frame)
        self.image_frame_layout.addWidget(scroll_area)
        
        # Right panel - Controls
        self.control_frame = QFrame()
        self.control_frame.setFixedWidth(400)
        self.control_frame.setStyleSheet("background-color: #f8f8f8;")
        
        self.control_layout = QVBoxLayout(self.control_frame)
        
        # Operation selection
        self.operation_combo = QComboBox()
        self.operation_combo.addItems([
            "Select Operation",
            "Grayscale",
            "Blur",
            "Edge Detection",
            "Sharpen",
            "Emboss",
            "Sepia",
            "Cartoon Effect",
            "Pencil Sketch",
            "Oil Painting",
            "Rotate",
            "Flip",
            "Resize",
            "Crop",
            "Brightness/Contrast",
            "Color Balance",
            "Threshold",
            "Histogram Equalization",
            "Watermark",
            "Text Overlay"
        ])
        self.operation_combo.currentIndexChanged.connect(self.show_operation_controls)
        
        # Operation parameters frame
        self.params_frame = QFrame()
        self.params_frame.setStyleSheet("background-color: #f8f8f8;")
        self.params_layout = QVBoxLayout(self.params_frame)
        
        # Action buttons
        self.button_layout = QHBoxLayout()
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self.open_image)
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_operation)
        self.apply_btn.setEnabled(False)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_image)
        self.reset_btn.setEnabled(False)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        
        self.button_layout.addWidget(self.open_btn)
        self.button_layout.addWidget(self.apply_btn)
        self.button_layout.addWidget(self.reset_btn)
        self.button_layout.addWidget(self.save_btn)
        
        # Add widgets to control layout
        self.control_layout.addWidget(QLabel("Select Operation:"))
        self.control_layout.addWidget(self.operation_combo)
        self.control_layout.addWidget(self.params_frame)
        self.control_layout.addLayout(self.button_layout)
        self.control_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Add frames to main layout
        self.layout.addWidget(self.image_frame)
        self.layout.addWidget(self.control_frame)
    
    def open_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        
        if filename:
            self.filename = filename
            self.image = cv2.imread(filename)
            self.original_image = self.image.copy()
            self.display_image()
            self.reset_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
    
    def display_image(self):
        if self.image is not None:
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.display_image()
    
    def show_operation_controls(self):
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        operation = self.operation_combo.currentText()
        
        if operation == "Select Operation":
            self.apply_btn.setEnabled(False)
            return
        
        self.apply_btn.setEnabled(True)
        
        if operation == "Blur":
            self.create_slider_control("Kernel Size", 1, 31, 5, 2, "blur_kernel")
        elif operation == "Edge Detection":
            self.create_dropdown_control("Method", ["Canny", "Sobel", "Laplacian"], "edge_method")
            self.create_slider_control("Threshold 1", 0, 255, 100, 1, "edge_thresh1")
            self.create_slider_control("Threshold 2", 0, 255, 200, 1, "edge_thresh2")
        elif operation == "Sharpen":
            self.create_slider_control("Strength", 1, 10, 3, 1, "sharpen_strength")
        elif operation == "Rotate":
            self.create_slider_control("Angle", -180, 180, 0, 1, "rotate_angle")
        elif operation == "Resize":
            self.create_slider_control("Width %", 10, 500, 100, 1, "resize_width")
            self.create_slider_control("Height %", 10, 500, 100, 1, "resize_height")
        elif operation == "Brightness/Contrast":
            self.create_slider_control("Brightness", -100, 100, 0, 1, "brightness")
            self.create_slider_control("Contrast", -100, 100, 0, 1, "contrast")
        elif operation == "Color Balance":
            self.create_slider_control("Red", -100, 100, 0, 1, "red_balance")
            self.create_slider_control("Green", -100, 100, 0, 1, "green_balance")
            self.create_slider_control("Blue", -100, 100, 0, 1, "blue_balance")
        elif operation == "Threshold":
            self.create_dropdown_control("Type", ["Binary", "Binary Inv", "Trunc", "To Zero", "To Zero Inv"], "thresh_type")
            self.create_slider_control("Threshold", 0, 255, 127, 1, "thresh_value")
        elif operation == "Watermark":
            self.create_entry_control("Watermark Text", "watermark_text")
            self.create_slider_control("Opacity", 0, 100, 50, 1, "watermark_opacity")
            self.create_dropdown_control("Position", ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"], "watermark_pos")
        elif operation == "Text Overlay":
            self.create_entry_control("Text", "text_content")
            self.create_slider_control("Font Size", 10, 100, 30, 1, "text_size")
            self.create_color_chooser("Text Color", "text_color")
            self.create_dropdown_control("Position", ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"], "text_pos")

    def create_slider_control(self, label, min_val, max_val, default, step, param_name):
        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)
        layout = QHBoxLayout(frame)
        
        label_widget = QLabel(label)
        layout.addWidget(label_widget)
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setSingleStep(step)
        slider.setPageStep(step * 5)
        layout.addWidget(slider)
        
        value_label = QLabel(str(default))
        layout.addWidget(value_label)
        
        slider.valueChanged.connect(lambda val: value_label.setText(str(val)))
        setattr(self, param_name, slider)
        self.params_layout.addWidget(frame)
    
    def create_dropdown_control(self, label, options, param_name):
        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)
        layout = QHBoxLayout(frame)
        
        label_widget = QLabel(label)
        layout.addWidget(label_widget)
        
        combo = QComboBox()
        combo.addItems(options)
        layout.addWidget(combo)
        
        setattr(self, param_name, combo)
        self.params_layout.addWidget(frame)
    
    def create_entry_control(self, label, param_name):
        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)
        layout = QHBoxLayout(frame)
        
        label_widget = QLabel(label)
        layout.addWidget(label_widget)
        
        line_edit = QLineEdit()
        layout.addWidget(line_edit)
        
        setattr(self, param_name, line_edit)
        self.params_layout.addWidget(frame)
    
    def create_color_chooser(self, label, param_name):
        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)
        layout = QHBoxLayout(frame)
        
        label_widget = QLabel(label)
        layout.addWidget(label_widget)
        
        line_edit = QLineEdit("#FFFFFF")
        layout.addWidget(line_edit)
        
        color_btn = QPushButton("...")
        color_btn.setFixedWidth(30)
        color_btn.clicked.connect(lambda: self.choose_color(line_edit))
        layout.addWidget(color_btn)
        
        setattr(self, param_name, line_edit)
        self.params_layout.addWidget(frame)
    
    def choose_color(self, line_edit):
        color = QColorDialog.getColor()
        if color.isValid():
            line_edit.setText(color.name())
    
    def apply_operation(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first")
            return
            
        operation = self.operation_combo.currentText()
        
        try:
            if operation == "Grayscale":
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            
            elif operation == "Blur":
                kernel_size = self.blur_kernel.value()
                kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
                self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
            
            elif operation == "Edge Detection":
                method = self.edge_method.currentText()
                thresh1 = self.edge_thresh1.value()
                thresh2 = self.edge_thresh2.value()
                
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                
                if method == "Canny":
                    edges = cv2.Canny(gray, thresh1, thresh2)
                elif method == "Sobel":
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
                    edges = cv2.magnitude(sobelx, sobely)
                    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                elif method == "Laplacian":
                    edges = cv2.Laplacian(gray, cv2.CV_64F)
                    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                
                self.image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            elif operation == "Sharpen":
                strength = self.sharpen_strength.value()
                kernel = np.array([[-1, -1, -1],
                                  [-1, 9+strength, -1],
                                  [-1, -1, -1]])
                self.image = cv2.filter2D(self.image, -1, kernel)
            
            elif operation == "Emboss":
                kernel = np.array([[0, -1, -1],
                                  [1, 0, -1],
                                  [1, 1, 0]])
                self.image = cv2.filter2D(self.image, -1, kernel)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            
            elif operation == "Sepia":
                kernel = np.array([[0.272, 0.534, 0.131],
                                  [0.349, 0.686, 0.168],
                                  [0.393, 0.769, 0.189]])
                self.image = cv2.transform(self.image, kernel)
                self.image = np.clip(self.image, 0, 255).astype(np.uint8)
            
            elif operation == "Cartoon Effect":
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 7)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                            cv2.THRESH_BINARY, 9, 9)
                color = cv2.bilateralFilter(self.image, 9, 300, 300)
                cartoon = cv2.bitwise_and(color, color, mask=edges)
                self.image = cartoon
            
            elif operation == "Pencil Sketch":
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                inv_gray = 255 - gray
                blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
                sketch = cv2.divide(gray, 255 - blur, scale=256)
                self.image = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
            
            elif operation == "Oil Painting":
                res = cv2.xphoto.oilPainting(self.image, 7, 1)
                self.image = res
            
            elif operation == "Rotate":
                angle = self.rotate_angle.value()
                h, w = self.image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                self.image = cv2.warpAffine(self.image, M, (w, h))
            
            elif operation == "Flip":
                self.image = cv2.flip(self.image, 1)
            
            elif operation == "Resize":
                width_percent = self.resize_width.value() / 100.0
                height_percent = self.resize_height.value() / 100.0
                new_width = int(self.image.shape[1] * width_percent)
                new_height = int(self.image.shape[0] * height_percent)
                self.image = cv2.resize(self.image, (new_width, new_height))
            
            elif operation == "Crop":
                h, w = self.image.shape[:2]
                crop_percent = 20
                x1 = int(w * crop_percent / 100)
                y1 = int(h * crop_percent / 100)
                x2 = int(w * (100 - crop_percent) / 100)
                y2 = int(h * (100 - crop_percent) / 100)
                self.image = self.image[y1:y2, x1:x2]
            
            elif operation == "Brightness/Contrast":
                brightness = self.brightness.value()
                contrast = self.contrast.value()
                self.image = cv2.convertScaleAbs(self.image, alpha=1 + contrast/100, beta=brightness)
            
            elif operation == "Color Balance":
                red = self.red_balance.value()
                green = self.green_balance.value()
                blue = self.blue_balance.value()
                b, g, r = cv2.split(self.image)
                r = cv2.add(r, red)
                g = cv2.add(g, green)
                b = cv2.add(b, blue)
                self.image = cv2.merge((b, g, r))
                self.image = np.clip(self.image, 0, 255)
            
            elif operation == "Threshold":
                thresh_type = self.thresh_type.currentText()
                thresh_value = self.thresh_value.value()
                
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                
                if thresh_type == "Binary":
                    _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
                elif thresh_type == "Binary Inv":
                    _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
                elif thresh_type == "Trunc":
                    _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_TRUNC)
                elif thresh_type == "To Zero":
                    _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_TOZERO)
                elif thresh_type == "To Zero Inv":
                    _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_TOZERO_INV)
                
                self.image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            elif operation == "Histogram Equalization":
                ycrcb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
                ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
                self.image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            
            elif operation == "Watermark":
                text = self.watermark_text.text()
                opacity = self.watermark_opacity.value() / 100.0
                position = self.watermark_pos.currentText()
                
                if not text:
                    QMessageBox.warning(self, "Warning", "Please enter watermark text")
                    return
                
                watermarked = self.image.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                thickness = 2
                
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                h, w = self.image.shape[:2]
                
                if position == "Top-Left":
                    pos = (10, text_size[1] + 10)
                elif position == "Top-Right":
                    pos = (w - text_size[0] - 10, text_size[1] + 10)
                elif position == "Bottom-Left":
                    pos = (10, h - 10)
                elif position == "Bottom-Right":
                    pos = (w - text_size[0] - 10, h - 10)
                else:  # Center
                    pos = ((w - text_size[0]) // 2, (h + text_size[1]) // 2)
                
                cv2.putText(watermarked, text, pos, font, font_scale, 
                           (255, 255, 255, 255), thickness, cv2.LINE_AA)
                self.image = cv2.addWeighted(self.image, 1 - opacity, watermarked, opacity, 0)
            
            elif operation == "Text Overlay":
                text = self.text_content.text()
                font_size = self.text_size.value()
                color = self.text_color.text()
                position = self.text_pos.currentText()
                
                if not text:
                    QMessageBox.warning(self, "Warning", "Please enter text")
                    return
                
                color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                color_bgr = color_rgb[::-1]
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = font_size / 30.0
                thickness = 2
                
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                h, w = self.image.shape[:2]
                
                if position == "Top-Left":
                    pos = (10, text_size[1] + 10)
                elif position == "Top-Right":
                    pos = (w - text_size[0] - 10, text_size[1] + 10)
                elif position == "Bottom-Left":
                    pos = (10, h - 10)
                elif position == "Bottom-Right":
                    pos = (w - text_size[0] - 10, h - 10)
                else:  # Center
                    pos = ((w - text_size[0]) // 2, (h + text_size[1]) // 2)
                
                cv2.putText(self.image, text, pos, font, font_scale, 
                           (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(self.image, text, pos, font, font_scale, 
                           color_bgr, thickness, cv2.LINE_AA)
            
            self.display_image()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def reset_image(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.display_image()
    
    def save_image(self):
        if self.image is None:
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", 
            os.path.basename(self.filename) if self.filename else "edited_image.jpg",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        
        if filename:
            try:
                cv2.imwrite(filename, self.image)
                QMessageBox.information(self, "Success", "Image saved successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

class StatisticsTab(QWidget):
    def __init__(self, color_counts, search_term, material_data=None, video_data=None):
        super().__init__()
        self.color_counts = color_counts
        self.search_term = search_term
        self.material_data = material_data or {}
        self.video_data = video_data or []
        
        self.layout = QVBoxLayout(self)
        
        # Create tabs for different statistics
        self.tabs = QTabWidget()
        
        # Color distribution tab
        self.color_tab = QWidget()
        self.color_layout = QVBoxLayout(self.color_tab)
        self.create_color_graph()
        self.tabs.addTab(self.color_tab, "Color Distribution")
        
        # Material types tab
        if self.material_data.get('types'):
            self.types_tab = QWidget()
            self.types_layout = QVBoxLayout(self.types_tab)
            self.create_types_graph()
            self.tabs.addTab(self.types_tab, "Material Types")
        
        # Price distribution tab
        if self.material_data.get('prices'):
            self.price_tab = QWidget()
            self.price_layout = QVBoxLayout(self.price_tab)
            self.create_price_graph()
            self.tabs.addTab(self.price_tab, "Price Distribution")
        
        # Video statistics tab
        if self.video_data:
            self.video_tab = QWidget()
            self.video_layout = QVBoxLayout(self.video_tab)
            self.create_video_graph()
            self.tabs.addTab(self.video_tab, "Video Stats")
        
        self.layout.addWidget(self.tabs)
    
    def create_color_graph(self):
        # Create matplotlib figure
        figure = Figure(figsize=(5, 4))
        canvas = FigureCanvas(figure)
        self.color_layout.addWidget(canvas)
        
        # Create network graph
        G = nx.Graph()
        nodes = [self.search_term] + list(self.color_counts.keys())
        
        # Add nodes and edges
        G.add_nodes_from(nodes)
        edges = [(node, self.search_term) for node in self.color_counts.keys()]
        G.add_edges_from(edges)
        
        # Draw on matplotlib canvas
        ax = figure.add_subplot(111)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=2000,
               node_color="skyblue", font_size=10, font_weight="bold")
        ax.set_title(f"Color Distribution for '{self.search_term}'")
        canvas.draw()
    
    def create_types_graph(self):
        figure = Figure(figsize=(5, 4))
        canvas = FigureCanvas(figure)
        self.types_layout.addWidget(canvas)
        
        ax = figure.add_subplot(111)
        types = list(self.material_data['types'].keys())
        counts = list(self.material_data['types'].values())
        
        ax.bar(types, counts, color='lightgreen')
        ax.set_title("Material Types Distribution")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
        canvas.draw()
    
    def create_price_graph(self):
        figure = Figure(figsize=(5, 4))
        canvas = FigureCanvas(figure)
        self.price_layout.addWidget(canvas)
        
        ax = figure.add_subplot(111)
        prices = self.material_data['prices']
        
        ax.hist(prices, bins=10, color='salmon', edgecolor='black')
        ax.set_title("Price Distribution")
        ax.set_xlabel("Price")
        ax.set_ylabel("Frequency")
        canvas.draw()
    
    def create_video_graph(self):
        figure = Figure(figsize=(5, 4))
        canvas = FigureCanvas(figure)
        self.video_layout.addWidget(canvas)
        
        ax = figure.add_subplot(111)
        views = [video['views'] for video in self.video_data]
        likes = [video['likes'] for video in self.video_data]
        indices = range(len(self.video_data))
        
        ax.bar(indices, views, color='lightblue', label='Views')
        ax.bar(indices, likes, color='orange', label='Likes', alpha=0.7)
        ax.set_title("Video Views and Likes")
        ax.set_xticks(indices)
        ax.set_xticklabels([f"Video {i+1}" for i in indices], rotation=45)
        ax.legend()
        canvas.draw()
class ScrapingThread(QThread):
    results_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, url):
        super().__init__()
        self.url = url
        
    def run(self):
        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            import time
            
            # Initialize the web driver
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")  # Run in background
            web = webdriver.Chrome(options=options)
            
            # Navigate to the URL
            web.get(self.url)
            
            # Wait for the page to load
            time.sleep(3)
            
            # Find elements by class name
            elements = web.find_elements(By.CLASS_NAME, 'cg-caption')
            
            # Extract text from elements
            results = [element.text for element in elements if element.text.strip()]
            
            # Close the web driver
            web.quit()
            
            # Emit results
            self.results_ready.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
class PackagingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        self.layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Packaging Materials Search")
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(header_label)
        
        # Search controls
        search_group = QGroupBox("Search Packaging Materials")
        search_layout = QVBoxLayout()
        
        # URL input field
        url_layout = QHBoxLayout()
        self.url_input = QLineEdit("https://www.uline.com/Cls_04/Boxes-Corrugated")
        self.url_input.setMinimumHeight(40)
        self.url_input.setFont(QFont("Arial", 12))
        url_layout.addWidget(QLabel("URL:"))
        url_layout.addWidget(self.url_input)
        
        # Search button
        self.scrape_button = QPushButton("Scrape Packaging Materials")
        self.scrape_button.setMinimumHeight(40)
        self.scrape_button.setFont(QFont("Arial", 12))
        self.scrape_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; }"
        )
        self.scrape_button.clicked.connect(self.scrape_packaging)
        
        search_layout.addLayout(url_layout)
        search_layout.addWidget(self.scrape_button)
        search_group.setLayout(search_layout)
        self.layout.addWidget(search_group)
        
        # Results area
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Arial", 11))
        self.results_text.setMinimumHeight(300)
        
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        self.layout.addWidget(results_group)
        
        # Status label with italic font fix
        self.status_label = QLabel("Ready to scrape packaging materials")
        italic_font = QFont("Arial", 10)
        italic_font.setItalic(True)
        self.status_label.setFont(italic_font)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)
        
    def scrape_packaging(self):
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Empty URL", "Please enter a URL to scrape.")
            return
            
        self.status_label.setText("Scraping in progress... Please wait.")
        self.results_text.clear()
        QApplication.processEvents()  # Update the UI
        
        try:
            # Create a thread to avoid freezing the UI
            self.scrape_thread = ScrapingThread(url)
            self.scrape_thread.results_ready.connect(self.update_results)
            self.scrape_thread.error_occurred.connect(self.handle_error)
            self.scrape_thread.start()
        except Exception as e:
            self.handle_error(str(e))
    
    def update_results(self, results):
        self.results_text.clear()
        
        if not results:
            self.results_text.setPlainText("No results found.")
            self.status_label.setText("Scraping completed. No results found.")
            return
            
        # Create a formatted list of results
        for i, item in enumerate(results, 1):
            self.results_text.append(f"{i}. {item}")
            
        self.status_label.setText(f"Scraping completed. Found {len(results)} packaging items.")
    
    def handle_error(self, error_message):
        self.results_text.setPlainText(f"Error during scraping: {error_message}")
        self.status_label.setText("Scraping failed.")


class CraftSearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Craft Materials Search with Image Editor")
        self.setGeometry(100, 100, 1200, 800)
    
    # Network manager for downloading images
        self.network_manager = QNetworkAccessManager()
        self.network_manager.finished.connect(self.on_image_downloaded)
    
    # SerpAPI configuration
        self.serpapi_key = "91ca7555e236f50257d83a40a091688265b7de4ee96dfe417dc9eb41c173a242"
    
    # Create tab widget
        self.tabs = QTabWidget()
    
    # Create tabs
        self.search_tab = self.create_search_tab()
        self.image_editor_tab = ImageEditorTab()
        self.packaging_tab = PackagingTab()  # New packaging tab
    
    # Add tabs
        self.tabs.addTab(self.search_tab, "Craft Search")
        self.tabs.addTab(self.image_editor_tab, "Image Editor")
        self.tabs.addTab(self.packaging_tab, "Packaging")  # Add the new tab
    
    # Set central widget
        self.setCentralWidget(self.tabs)
    
    # Set up status bar
        self.statusBar().showMessage("Ready")
    
    def create_search_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        
        # Header section
        header_label = QLabel("Craft Materials & Video Search")
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)
        
        # Search section
        search_group = QGroupBox("Search")
        search_layout = QVBoxLayout()
        
        # Material and craft input fields
        input_form = QFormLayout()
        input_form.setSpacing(10)
        
        self.material_input = QLineEdit()
        self.material_input.setPlaceholderText("Enter material (e.g., 'cotton', 'wood')")
        self.material_input.setMinimumHeight(40)
        self.material_input.setFont(QFont("Arial", 12))
        
        self.craft_input = QLineEdit()
        self.craft_input.setPlaceholderText("Enter craft (e.g., 'quilting', 'scrapbooking')")
        self.craft_input.setMinimumHeight(40)
        self.craft_input.setFont(QFont("Arial", 12))
        
        input_form.addRow("Material:", self.material_input)
        input_form.addRow("Craft:", self.craft_input)
        
        # Search button
        self.search_button = QPushButton("Search")
        self.search_button.setMinimumHeight(40)
        self.search_button.setFont(QFont("Arial", 12))
        self.search_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; }")
        self.search_button.clicked.connect(self.perform_search)
        
        # Connect return pressed for both inputs
        self.material_input.returnPressed.connect(self.perform_search)
        self.craft_input.returnPressed.connect(self.perform_search)
        
        # Add options
        options_layout = QHBoxLayout()
        
        # Material search options
        material_options = QGroupBox("Material Options")
        material_options_layout = QVBoxLayout()
        self.include_fabrics = QCheckBox("Include Fabrics")
        self.include_fabrics.setChecked(True)
        self.include_crafting = QCheckBox("Include Crafting Supplies")
        self.include_crafting.setChecked(True)
        material_options_layout.addWidget(self.include_fabrics)
        material_options_layout.addWidget(self.include_crafting)
        material_options.setLayout(material_options_layout)
        
        # Video search options
        video_options = QGroupBox("Video Options")
        video_options_layout = QVBoxLayout()
        self.include_tutorials = QCheckBox("Include Tutorials")
        self.include_tutorials.setChecked(True)
        self.include_reviews = QCheckBox("Include Reviews")
        self.include_reviews.setChecked(True)
        video_options_layout.addWidget(self.include_tutorials)
        video_options_layout.addWidget(self.include_reviews)
        video_options.setLayout(video_options_layout)
        
        options_layout.addWidget(material_options)
        options_layout.addWidget(video_options)
        
        search_layout.addLayout(input_form)
        search_layout.addWidget(self.search_button)
        search_layout.addLayout(options_layout)
        search_group.setLayout(search_layout)
        
        # Create tabs for different results
        self.results_tabs = QTabWidget()
        self.materials_tab = QWidget()
        self.videos_tab = QWidget()
        
        # Materials tab layout
        materials_layout = QVBoxLayout()
        self.materials_scroll = QScrollArea()
        self.materials_scroll.setWidgetResizable(True)
        self.materials_content = QWidget()
        self.materials_grid = QGridLayout(self.materials_content)
        self.materials_grid.setSpacing(10)
        self.materials_scroll.setWidget(self.materials_content)
        materials_layout.addWidget(self.materials_scroll)
        self.materials_tab.setLayout(materials_layout)
        
        # Videos tab layout
        videos_layout = QVBoxLayout()
        self.videos_scroll = QScrollArea()
        self.videos_scroll.setWidgetResizable(True)
        self.videos_content = QWidget()
        self.videos_grid = QVBoxLayout(self.videos_content)
        self.videos_grid.setSpacing(10)
        self.videos_scroll.setWidget(self.videos_content)
        videos_layout.addWidget(self.videos_scroll)
        self.videos_tab.setLayout(videos_layout)
        
        # Add tabs to tab widget
        self.results_tabs.addTab(self.materials_tab, "Materials")
        self.results_tabs.addTab(self.videos_tab, "YouTube Videos")
        
        # Add all widgets to main layout
        main_layout.addWidget(search_group)
        main_layout.addWidget(self.results_tabs)
        
        return tab
    
    def perform_search(self):
        material = self.material_input.text().strip()
        craft = self.craft_input.text().strip()

        if not material:
            QMessageBox.warning(self, "Empty Search", "Please enter a material to search for.")
            return
        
        self.statusBar().showMessage(f"Searching for '{material}'...")
        
        # Clear previous results
        self.clear_results()
        
        # Perform search and get color_counts
        url = f"https://fabricla.com/search?q={material}&options%5Bprefix%5D=last"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch search results: {str(e)}")
            return
        
        # Find all product containers   
        product_containers = soup.find_all('div', class_="grid-product__content")
        
        # Initialize data collection structures
        color_counts = {}
        material_data = {
            'types': {},  # Material types and their counts
            'prices': [],  # List of all prices
            'materials': []  # List of material names for word cloud
        }
        video_data = []
        
        # Process each product
        for container in product_containers:
            # Extract color information
            title_div = container.find('div', class_="grid-product__title grid-product__title--heading")
            if title_div:
                result_text = title_div.text.strip()
                color_keywords = ["Baby Pink", "Neon Pink", "Hot Pink", "Dust Pink", "Pink", 
                                "Blue", "Red", "Green", "Yellow", "Black", "White"]
                found_color = None
                
                for color in color_keywords:
                    if color in result_text:
                        found_color = color
                        break

                if found_color:
                    color_counts[found_color] = color_counts.get(found_color, 0) + 1
                else:
                    color_counts["Other"] = color_counts.get("Other", 0) + 1
            
            # Extract material type and price
            material_type = "Unknown"
            price = 0.0
            
            # Try to extract material type from title
            if title_div:
                material_name = title_div.text.strip()
                material_data['materials'].append(material_name)
                
                # Simple material type detection (customize as needed)
                if "cotton" in material_name.lower():
                    material_type = "Cotton"
                elif "wool" in material_name.lower():
                    material_type = "Wool"
                elif "silk" in material_name.lower():
                    material_type = "Silk"
                else:
                    material_type = "Other Fabric"
            
            # Extract price
            price_div = container.find('div', class_="grid-product__price")
            if price_div:
                try:
                    price_text = price_div.text.strip()
                    price = float(''.join(c for c in price_text if c.isdigit() or c == '.'))
                    material_data['prices'].append(price)
                except ValueError:
                    pass
            
            # Update material type counts
            material_data['types'][material_type] = material_data['types'].get(material_type, 0) + 1
        
        # Get YouTube video data if enabled
        if self.include_tutorials.isChecked() or self.include_reviews.isChecked():
            try:
                params = {
                    "engine": "google_videos",
                    "q": f"{material} {craft}" if craft else material,
                    "api_key": self.serpapi_key
                }
                
                if self.include_tutorials.isChecked() and not self.include_reviews.isChecked():
                    params["q"] += " tutorial"
                elif not self.include_tutorials.isChecked() and self.include_reviews.isChecked():
                    params["q"] += " review"
                
                search = GoogleSearch(params)
                results = search.get_dict()
                
                for video in results.get("video_results", []):
                    video_data.append({
                        'title': video.get('title', 'No title'),
                        'views': self.parse_video_views(video.get('views', '0')),
                        'likes': random.randint(100, 10000),  # Placeholder - real API would provide this
                        'duration': video.get('duration', 'N/A'),
                        'channel': video.get('channel', {}).get('name', 'Unknown')
                    })
            except Exception as e:
                print(f"Error fetching video data: {e}")
        
        # Add/update statistics tab with all collected data
        if hasattr(self, 'stats_tab') and self.stats_tab:
            self.tabs.removeTab(self.tabs.indexOf(self.stats_tab))
            
        self.stats_tab = StatisticsTab(
            color_counts, 
            material,
            material_data=material_data,
            video_data=video_data if video_data else None
        )
        self.tabs.addTab(self.stats_tab, "Statistics")
        
        # Perform materials and videos search
        if self.include_fabrics.isChecked() or self.include_crafting.isChecked():
            self.search_materials(material, craft)
        
        if self.include_tutorials.isChecked() or self.include_reviews.isChecked():
            self.search_youtube_videos(material, craft)
        
        self.statusBar().showMessage(f"Search completed for '{material}'")
    
    def parse_video_views(self, view_str):
        """Convert view count strings (e.g., '1.2M views') to integers"""
        if not view_str:
            return 0
        try:
            view_str = view_str.lower().replace('views', '').strip()
            if 'k' in view_str:
                return int(float(view_str.replace('k', '')) * 1000)
            elif 'm' in view_str:
                return int(float(view_str.replace('m', '')) * 1000000)
            return int(''.join(c for c in view_str if c.isdigit()))
        except:
            return 0
    
    def clear_results(self):
        # Clear materials grid
        while self.materials_grid.count():
            item = self.materials_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        # Clear videos layout
        while self.videos_grid.count():
            item = self.videos_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def search_materials(self, material, craft):
        try:
            # Build search query based on options
            query = material
            if craft:
                query += f" for {craft}"
                
            if self.include_fabrics.isChecked() and not self.include_crafting.isChecked():
                query += " fabric"
            elif not self.include_fabrics.isChecked() and self.include_crafting.isChecked():
                query += " craft supplies"
                
            url = f"https://fabricla.com/search?q={query}&options%5Bprefix%5D=last"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                self.materials_grid.addWidget(QLabel(f"Error: Could not access FabricLA (Status code: {response.status_code})"), 0, 0)
                return
                
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find product containers
            product_containers = soup.find_all('div', class_="grid-product__content")
            
            if not product_containers:
                self.materials_grid.addWidget(QLabel(f"No materials found for {material} in {craft if craft else 'crafting'}."), 0, 0)
                return
                
            # Add result count
            count_label = QLabel(f"Found {len(product_containers)} materials matching '{material}' for {craft if craft else 'crafting'}")
            count_label.setFont(QFont("Arial", 10, QFont.Bold))
            self.materials_grid.addWidget(count_label, 0, 0, 1, 2)
            
            for i, container in enumerate(product_containers):
                # Get material name
                title_div = container.find('div', class_="grid-product__title grid-product__title--heading")
                if not title_div:
                    continue
                material_name = title_div.text.strip()
                
                # Find image element
                image_element = container.find('image-element')
                img_tag = image_element.find('img') if image_element else None
                
                # Create result card
                card = QFrame()
                card.setFrameShape(QFrame.Box)
                card.setLineWidth(1)
                card.setMidLineWidth(0)
                card.setStyleSheet("""
                    QFrame { 
                        background-color: #f9f9f9; 
                        border-radius: 8px; 
                        border: 1px solid #ddd;
                        padding: 10px;
                    }
                    QFrame:hover {
                        background-color: #f0f0f0;
                        border: 1px solid #bbb;
                    }
                """)
                
                card_layout = QVBoxLayout()
                
                # Create image label
                image_label = QLabel()
                image_label.setFixedSize(200, 150)
                image_label.setAlignment(Qt.AlignCenter)
                image_label.setStyleSheet("""
                    QLabel {
                        background-color: #eee;
                        border-radius: 4px;
                        border: 1px solid #ddd;
                    }
                """)
                image_label.setScaledContents(True)
                
                # Try to find and download the image
                if img_tag:
                    image_url = img_tag.get('data-src') or img_tag.get('src')
                    if image_url:
                        if image_url.startswith('//'):
                            image_url = 'https:' + image_url
                        elif image_url.startswith('/'):
                            image_url = 'https://fabricla.com' + image_url
                        
                        if '?' in image_url:
                            image_url = image_url.split('?')[0]
                        
                        request = QNetworkRequest(QUrl(image_url))
                        reply = self.network_manager.get(request)
                        reply.setProperty('image_label', image_label)
                    else:
                        image_label.setText("🧶")
                        image_label.setFont(QFont("Arial", 24))
                else:
                    image_label.setText("🧶")
                    image_label.setFont(QFont("Arial", 24))
                
                # Material name
                name_label = QLabel(material_name)
                name_label.setWordWrap(True)
                name_label.setFont(QFont("Arial", 10, QFont.Bold))
                name_label.setStyleSheet("color: #333;")
                
                # Price if available
                price_div = container.find('div', class_="grid-product__price")
                price_label = QLabel(price_div.text.strip() if price_div else "Price not available")
                price_label.setFont(QFont("Arial", 9))
                price_label.setStyleSheet("color: #00796b;")
                
                card_layout.addWidget(image_label)
                card_layout.addWidget(name_label)
                card_layout.addWidget(price_label)
                card.setLayout(card_layout)
                
                # Add to grid - 2 columns
                row = (i // 2) + 1
                col = i % 2
                self.materials_grid.addWidget(card, row, col)
                
            self.materials_grid.setRowStretch(self.materials_grid.rowCount(), 1)
            
        except Exception as e:
            self.materials_grid.addWidget(QLabel(f"Error searching materials: {str(e)}"), 0, 0)
    
    def search_youtube_videos(self, material, craft):
        try:
            # Build search query based on options
            query = f"{material}"
            if craft:
                query += f" for {craft}"
            
            if self.include_tutorials.isChecked() and not self.include_reviews.isChecked():
                query += " tutorial"
            elif not self.include_tutorials.isChecked() and self.include_reviews.isChecked():
                query += " review"
            
            # Make API request to SerpAPI
            params = {
                "engine": "google_videos",
                "q": query,
                "api_key": self.serpapi_key
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Extract video results
            videos = results.get("video_results", [])
            
            if not videos:
                self.videos_grid.addWidget(QLabel(f"No videos found for '{material}' in {craft if craft else 'crafting'}"))
                return
            
            # Add result count
            count_label = QLabel(f"Found {len(videos)} videos for '{material}' in {craft if craft else 'crafting'}")
            count_label.setFont(QFont("Arial", 10, QFont.Bold))
            self.videos_grid.addWidget(count_label)
            
            for video in videos:
                if 'link' not in video:
                    continue
                    
                # Create video result card
                card = QFrame()
                card.setFrameShape(QFrame.Box)
                card.setLineWidth(1)
                card.setStyleSheet("""
                    QFrame { 
                        background-color: #f0f0f0; 
                        border-radius: 8px; 
                        margin: 5px;
                        border: 1px solid #ddd;
                        padding: 10px;
                    }
                    QFrame:hover {
                        background-color: #e6e6e6;
                        border: 1px solid #bbb;
                    }
                """)
                
                card_layout = QHBoxLayout()
                
                # Thumbnail
                thumbnail_label = QLabel()
                thumbnail_label.setFixedSize(120, 90)
                thumbnail_label.setAlignment(Qt.AlignCenter)
                thumbnail_label.setStyleSheet("""
                    QLabel {
                        background-color: #ddd;
                        border-radius: 4px;
                        border: 1px solid #bbb;
                    }
                """)
                thumbnail_label.setScaledContents(True)
                thumbnail_label.setProperty('is_video', True)
                
                if video.get('thumbnail'):
                    thumbnail_url = video['thumbnail']
                    request = QNetworkRequest(QUrl(thumbnail_url))
                    reply = self.network_manager.get(request)
                    reply.setProperty('image_label', thumbnail_label)
                else:
                    thumbnail_label.setText("▶️")
                    thumbnail_label.setFont(QFont("Arial", 24))
                
                # Content
                content_layout = QVBoxLayout()
                
                title = video.get('title', 'No title available')
                link = video['link']
                title_label = QLabel(f"<a href='{link}'>{title}</a>")
                title_label.setOpenExternalLinks(True)
                title_label.setWordWrap(True)
                title_label.setFont(QFont("Arial", 11, QFont.Bold))
                title_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
                
                details_layout = QHBoxLayout()
                channel_label = QLabel(f"Channel: {video.get('channel', {}).get('name', 'Unknown')}")
                channel_label.setFont(QFont("Arial", 9))
                
                duration_label = QLabel(f"Duration: {video.get('duration', 'N/A')}")
                duration_label.setFont(QFont("Arial", 9))
                
                details_layout.addWidget(channel_label)
                details_layout.addWidget(duration_label)
                details_layout.addStretch()
                
                position_label = QLabel(f"Rank: {video.get('position', 'N/A')}")
                position_label.setFont(QFont("Arial", 8))
                position_label.setStyleSheet("color: #666;")
                
                content_layout.addWidget(title_label)
                content_layout.addLayout(details_layout)
                content_layout.addWidget(position_label)
                
                card_layout.addWidget(thumbnail_label)
                card_layout.addLayout(content_layout)
                card.setLayout(card_layout)
                
                self.videos_grid.addWidget(card)
            
            self.videos_grid.addStretch()
            
        except Exception as e:
            error_label = QLabel(f"Error searching YouTube videos: {str(e)}")
            error_label.setWordWrap(True)
            self.videos_grid.addWidget(error_label)
    
    def on_image_downloaded(self, reply):
        # Get the image label associated with this reply
        image_label = reply.property('image_label')
        if not image_label:
            return
            
        if reply.error():
            # Set placeholder if download fails
            if isinstance(image_label.property('is_video'), bool) and image_label.property('is_video'):
                image_label.setText("▶️")
            else:
                image_label.setText("🧶")
            image_label.setFont(QFont("Arial", 24))
            return
            
        # Read the image data
        image_data = reply.readAll()
        image = QImage()
        image.loadFromData(image_data)
        
        if image.isNull():
            # Set placeholder if image is invalid
            if isinstance(image_label.property('is_video'), bool) and image_label.property('is_video'):
                image_label.setText("▶️")
            else:
                image_label.setText("🧶")
            image_label.setFont(QFont("Arial", 24))
        else:
            # Set the image to the label
            pixmap = QPixmap.fromImage(image)
            if isinstance(image_label.property('is_video'), bool) and image_label.property('is_video'):
                image_label.setPixmap(pixmap.scaled(120, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                image_label.setPixmap(pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CraftSearchApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()