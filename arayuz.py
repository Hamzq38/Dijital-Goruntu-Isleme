import sys
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame, QInputDialog, QMessageBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

class GoruntuIslemeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Görüntü İşleme Ödevi - Tam Sürüm")
        
        self.image_array = None
        self.processed_array = None

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setFixedWidth(250)
        left_layout = QVBoxLayout(left_panel)

        self.btn_open = QPushButton("📂 Dosya Seç (Open File)")
        self.btn_save = QPushButton("💾 Kaydet (Save As)")
        self.btn_gray = QPushButton("1. Gri Seviyeye Çevir")
        self.btn_yuv = QPushButton("2. YUV Kanallarını Göster")
        self.btn_binary = QPushButton("3. Binary (Siyah-Beyaz)")
        self.btn_hist_draw = QPushButton("4. Histogram Çiz")
        self.btn_hist_eq = QPushButton("5. Histogram Eşitleme")
        self.btn_contrast = QPushButton("6. Kontrast Germe")
        
        btn_style = "padding: 8px; font-weight: bold; background-color: #f0f0f0; color: black; margin-bottom: 5px;"
        for btn in [self.btn_open, self.btn_save, self.btn_gray, self.btn_yuv, 
                    self.btn_binary, self.btn_hist_draw, self.btn_hist_eq, self.btn_contrast]:
            btn.setStyleSheet(btn_style)
            left_layout.addWidget(btn)

        self.btn_open.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_gray.clicked.connect(self.apply_grayscale)
        self.btn_yuv.clicked.connect(self.apply_yuv)
        self.btn_binary.clicked.connect(self.apply_binary)
        self.btn_hist_draw.clicked.connect(self.draw_histogram)
        self.btn_hist_eq.clicked.connect(self.histogram_equalization)
        self.btn_contrast.clicked.connect(self.contrast_stretching)

        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QHBoxLayout(right_panel)

        self.lbl_original = QLabel("Orijinal Görüntü\n(Henüz seçilmedi)")
        self.lbl_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_original.setStyleSheet("border: 2px dashed #aaa; background-color: #e0e0e0; color: black; font-size: 16px;")
        
        self.lbl_processed = QLabel("İşlenmiş Görüntü\n(İşlem bekleniyor)")
        self.lbl_processed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_processed.setStyleSheet("border: 2px dashed #aaa; background-color: #e0e0e0; color: black; font-size: 16px;")

        right_layout.addWidget(self.lbl_original)
        right_layout.addWidget(self.lbl_processed)

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 4)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Görüntü Seç", "", "Görüntü Dosyaları (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            pil_image = Image.open(file_name).convert("RGB")
            self.image_array = np.array(pil_image)
            self.display_image(self.image_array, self.lbl_original)
            self.lbl_processed.setText("İşlem bekleniyor...")

    def save_image(self):
        if self.processed_array is not None:
            file_name, _ = QFileDialog.getSaveFileName(self, "Görüntüyü Kaydet", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_name:
                Image.fromarray(self.processed_array).save(file_name)

    def get_gray_array(self):
        R = self.image_array[:, :, 0]
        G = self.image_array[:, :, 1]
        B = self.image_array[:, :, 2]
        return np.clip(0.299 * R + 0.587 * G + 0.114 * B, 0, 255).astype(np.uint8)

    def check_image(self):
        if self.image_array is None:
            QMessageBox.warning(self, "Hata", "Lütfen önce bir görüntü seçin!")
            return False
        return True

    def apply_grayscale(self):
        if not self.check_image(): return
        self.processed_array = self.get_gray_array()
        self.display_image(self.processed_array, self.lbl_processed)

    def apply_yuv(self):
        if not self.check_image(): return
        
        R = self.image_array[:, :, 0]
        G = self.image_array[:, :, 1]
        B = self.image_array[:, :, 2]

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = 0.492 * (B - Y)
        V = 0.877 * (R - Y)

        Y_img = np.clip(Y, 0, 255).astype(np.uint8)
        U_img = np.clip(U + 128, 0, 255).astype(np.uint8) 
        V_img = np.clip(V + 128, 0, 255).astype(np.uint8)

        self.processed_array = np.hstack((Y_img, U_img, V_img))
        self.display_image(self.processed_array, self.lbl_processed)

    def apply_binary(self):
        if not self.check_image(): return
        threshold, ok = QInputDialog.getInt(self, "Binary Dönüşüm", "Eşik Değeri (0-255):", 128, 0, 255)
        if ok:
            gray = self.get_gray_array()
            self.processed_array = np.where(gray >= threshold, 255, 0).astype(np.uint8)
            self.display_image(self.processed_array, self.lbl_processed)

    def draw_histogram(self):
        if not self.check_image(): return
        gray = self.get_gray_array()
        
        histogram = np.bincount(gray.flatten(), minlength=256)
        
        hist_img = np.ones((256, 256), dtype=np.uint8) * 255 
        
        max_val = histogram.max()
        if max_val > 0:
            hist_norm = (histogram * 255 / max_val).astype(int) 
            
            for x in range(256):
                h = hist_norm[x]
                if h > 0:
                    hist_img[256-h:, x] = 0 
                    
        self.processed_array = hist_img
        self.display_image(self.processed_array, self.lbl_processed)

    def histogram_equalization(self):
        if not self.check_image(): return
        gray = self.get_gray_array()
        
        histogram = np.bincount(gray.flatten(), minlength=256)
        cdf = histogram.cumsum() 
        
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_final = np.ma.filled(cdf_masked, 0).astype(np.uint8)
        
        self.processed_array = cdf_final[gray]
        self.display_image(self.processed_array, self.lbl_processed)

    def contrast_stretching(self):
        if not self.check_image(): return
        min_val, ok1 = QInputDialog.getInt(self, "Kontrast Germe", "Alt Sınır (0-255):", 50, 0, 255)
        if ok1:
            max_val, ok2 = QInputDialog.getInt(self, "Kontrast Germe", "Üst Sınır (0-255):", 200, 0, 255)
            if ok2 and min_val < max_val:
                gray = self.get_gray_array()
                
                stretched = (gray.astype(float) - min_val) * (255.0 / (max_val - min_val))
                
                self.processed_array = np.clip(stretched, 0, 255).astype(np.uint8)
                self.display_image(self.processed_array, self.lbl_processed)

    def display_image(self, img_array, label):
        if len(img_array.shape) == 3:
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        elif len(img_array.shape) == 2:
            height, width = img_array.shape
            bytes_per_line = width
            q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GoruntuIslemeApp()
    ex.resize(1200, 650)
    ex.show()
    sys.exit(app.exec())