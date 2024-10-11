# -*- coding: utf-8 -*-
# pyinstaller -w hailing_1.2.py --paths="C:\Users\chenwenqiang\anaconda3\envs\pytorch\Lib\site-packages\cv2"
# 317白
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# 捕获画面模糊检测类
class BlurDetection:
    def __init__(self):
        self.laplacian_scores = []
        self.brenner_scores = []
        self.energy_gradient_scores = []

        # 阈值
        self.energy_gradient_threshold = 44800 #33150
        self.laplacian_threshold = 3000 #3240
        self.brenner_threshold = 960 #711

    # Laplacian算子法
    def calculate_laplacian_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()  # 计算图像的拉普拉斯方差
        self.laplacian_scores.append(laplacian)  # 将当前帧的拉普拉斯方差值添加到列表中
    # Brenner算法
    def calculate_brenner_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)  # 将彩色图像转换为灰度图像，并转换为float32类型
        height, width = gray.shape[:2]  # 获取灰度图像的高度和宽度
        brenner_diff = np.diff(gray[:, 2:], axis=1) ** 2  # 计算沿水平方向的像素差值的平方
        brenner_sum = np.sum(brenner_diff)  # 将所有像素差值的平方求和
        brenner = brenner_sum / ((height - 2) * (width - 2))   # 计算Brenner得分，除以有效像素点的数量
        self.brenner_scores.append(brenner)
    # 能量梯度法
    def calculate_energy_gradient_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像
        height, width = gray.shape[:2]  # 获取灰度图像的高度和宽度
        resized_gray = cv2.resize(gray, (width - 1, height - 1))  # 将灰度图像缩小一行一列，用于计算梯度
        gradient_x = cv2.Sobel(resized_gray, cv2.CV_64F, 1, 0, ksize=3)  # 计算X方向上的梯度
        gradient_y = cv2.Sobel(resized_gray, cv2.CV_64F, 0, 1, ksize=3)  # 计算Y方向上的梯度
        energy_gradient = gradient_x ** 2 + gradient_y ** 2  # 计算能量梯度得分，即梯度的平方和
        energy_gradient_sum = np.sum(energy_gradient)  # 计算能量梯度得分的总和
        num_pixels = (height - 1) * (width - 1)  # 计算有效像素点的数量
        energy_gradient = energy_gradient_sum / num_pixels  # 计算能量梯度得分，即梯度的平方和除以有效像素点的数量
        self.energy_gradient_scores.append(energy_gradient)
    def is_laplacian_pass(self):
        current_score = self.laplacian_scores[-1]
        if current_score >= self.laplacian_threshold:
            return "合格"
        else:
            return "不良"
    # 判断Brenner评分是否合格
    def is_brenner_pass(self):
        current_score = self.brenner_scores[-1]
        if current_score >= self.brenner_threshold:
            return "合格"
        else:
            return "不良"
    # 判断能量梯度法评分是否合格
    def is_energy_gradient_pass(self):
        current_score = self.energy_gradient_scores[-1]
        if current_score >= self.energy_gradient_threshold:
            return "合格"
        else:
            return "不良"
# 创建UI窗口
class App:
    def __init__(self, root):
        self.root = root
        self.video_source = 0
        self.video_capture = None
        self.blur_detection = BlurDetection()
        self.camera_opened = False # 摄像头状态，默认关闭
        # 创建窗口组件
        self.canvas = tk.Canvas(root, width=960, height=768)#869,700
        self.canvas.grid(row=0, column=0, rowspan=28, padx=10, pady=10)
        # 数值标签
        self.laplacian_label = tk.Label(root, text="Laplacian: ", font=("Helvetica", 14))
        self.brenner_label = tk.Label(root, text="Brenner: ", font=("Helvetica", 14))
        self.energy_gradient_label = tk.Label(root, text="E_g: ", font=("Helvetica", 14))

        self.laplacian_test_label = tk.Label(root, text="L测试: ", font=("Helvetica", 14))
        self.brenner_test_label = tk.Label(root, text="B测试: ", font=("Helvetica", 14))
        self.energy_gradient_test_label = tk.Label(root, text="E测试: ", font=("Helvetica", 14))

        self.E_threshold_label = tk.Label(root, text="E阈值: ",font=("Helvetica", 14))
        self.L_threshold_label = tk.Label(root, text="L阈值: ",font=("Helvetica", 14))
        self.B_threshold_label = tk.Label(root, text="B阈值: ",font=("Helvetica", 14))

        self.camera_button = tk.Button(root, text="打开摄像头", command=self.toggle_camera, font=("Helvetica", 14))
        self.camera_button.grid(row=1, column=4, pady=10)

        self.energy_gradient_label.grid(row=4, column=4, sticky="w", pady=(0, 5))
        self.energy_gradient_test_label.grid(row=5, column=4, sticky="w", pady=(0, 5))
        self.E_threshold_label.grid(row=6, column=4, sticky="w", pady=(0, 5))

        self.laplacian_label.grid(row=7, column=4, sticky="w", pady=(0, 5))
        self.laplacian_test_label.grid(row=8, column=4, sticky="w", pady=(0, 5))
        self.L_threshold_label.grid(row=9, column=4, sticky="w", pady=(0, 5))

        self.brenner_label.grid(row=10, column=4, sticky="w", pady=(0, 5))
        self.brenner_test_label.grid(row=11, column=4, sticky="w", pady=(0, 5))
        self.B_threshold_label.grid(row=12, column=4, sticky="w", pady=(0, 5))
    # 切换摄像头状态
    def toggle_camera(self):
        if self.camera_opened:
            self.close_camera()
        else:
            self.open_camera()
    # 打开摄像头
    def open_camera(self):
        self.video_capture = cv2.VideoCapture(1)
        self.camera_opened = True
        self.camera_button.configure(text="关闭摄像头")
        self.update_image()

    # 从摄像头获取画面并更新到窗口中
    def update_image(self):
        ret, frame = self.video_capture.read()
        if ret:
            height, width, _ = frame.shape
            # 计算矩形的中心点坐标
            center_x = width // 2
            center_y = height // 2

            # 计算矩形的宽度和高度
            rect_width = int(width * 0.6)  # 矩形的宽度
            rect_height = int(height * 0.6)  # 矩形的高度
            # 绘制矩形区域为白色
            top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
            bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 1)
            # 创建一个和图像大小相同的黑色掩码
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, top_left, bottom_right, 255, -1)
            # 提取矩形区域内的图像
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            self.blur_detection.calculate_laplacian_score(masked_frame)
            self.blur_detection.calculate_brenner_score(masked_frame)
            self.blur_detection.calculate_energy_gradient_score(masked_frame)

            energy_gradient_str = f"E_g: {self.blur_detection.energy_gradient_scores[-1]:.2f}"
            if self.blur_detection.energy_gradient_scores[-1] < 600:
                self.blur_detection.energy_gradient_max = 0
            e_threshold_str = f"E阈值: {self.blur_detection.energy_gradient_threshold:.2f}"
            self.energy_gradient_label.configure(text=energy_gradient_str)
            self.E_threshold_label.configure(text=e_threshold_str)

            laplacian_str = f"Laplacian: {self.blur_detection.laplacian_scores[-1]:.2f}"
            if self.blur_detection.laplacian_scores[-1] < 200:
                self.blur_detection.laplacian_max = 0
            l_threshold_str = f"L阈值: {self.blur_detection.laplacian_threshold:.2f}"
            self.laplacian_label.configure(text=laplacian_str)
            self.L_threshold_label.configure(text=l_threshold_str)

            brenner_str = f"Brenner: {self.blur_detection.brenner_scores[-1]:.2f}"
            if self.blur_detection.brenner_scores[-1] < 50:
                self.blur_detection.brenner_max = 0
            b_threshold_str = f"B阈值: {self.blur_detection.brenner_threshold:.2f}"
            self.brenner_label.configure(text=brenner_str)
            self.B_threshold_label.configure(text=b_threshold_str)

            laplacian_pass_str = self.blur_detection.is_laplacian_pass()
            brenner_pass_str = self.blur_detection.is_brenner_pass()
            energy_gradient_pass_str = self.blur_detection.is_energy_gradient_pass()

            self.laplacian_test_label.configure(text=f"L测试：{laplacian_pass_str}",fg="green" if laplacian_pass_str == "合格" else "red")
            self.brenner_test_label.configure(text=f"B测试：{brenner_pass_str}",fg="green" if brenner_pass_str == "合格" else "red")
            self.energy_gradient_test_label.configure(text=f"E测试：{energy_gradient_pass_str}",fg="green" if energy_gradient_pass_str == "合格" else "red")

            # 混合原始画面和圆形区域内的画面
            blended_frame = cv2.addWeighted(masked_frame, 0.3, frame, 0.7, 0)
            # 放大
            enlarged_frame = cv2.resize(blended_frame, (960, 768))  # 修改放大尺寸960,768，1360*768-613*492
            # 在图片中心添加红色十字架
            h, w, _ = enlarged_frame.shape
            crosshair_size = 30
            crosshair_thickness = 1
            crosshair_color = (0, 0, 255)  # 红色
            x_center = w // 2
            y_center = h // 2
            cv2.drawMarker(enlarged_frame, (x_center, y_center), crosshair_color, markerType=cv2.MARKER_CROSS,
                           markerSize=crosshair_size, thickness=crosshair_thickness)
            enlarged_image = Image.fromarray(cv2.cvtColor(enlarged_frame, cv2.COLOR_BGR2RGB))
            enlarged_photo = ImageTk.PhotoImage(enlarged_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=enlarged_photo)
            self.canvas.image = enlarged_photo
        self.root.after(20, self.update_image)

    # 关闭摄像头
    def close_camera(self):
        if self.video_capture is not None:
            self.canvas.delete("all")
            self.video_capture.release()
            self.video_capture = None
        self.camera_opened = False
        self.calibration_done = False
        self.camera_button.configure(text="打开摄像头")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("白色317-模糊要拔插")
    app = App(root)
    root.mainloop()