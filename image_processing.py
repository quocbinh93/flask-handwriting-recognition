import cv2
import numpy as np

def preprocess_image(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)

    # Chuyển đổi ảnh sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Làm mịn ảnh bằng Gaussian Blur (giảm nhiễu)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Tìm ngưỡng động dựa trên giá trị trung bình của ảnh
    _, bin_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Tìm tất cả các contours
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc contours dựa trên diện tích và tỷ lệ khung hình
    area_threshold = 50  # Điều chỉnh ngưỡng diện tích tối thiểu
    aspect_ratio_threshold = 0.2  # Điều chỉnh ngưỡng tỷ lệ khung hình
    filtered_contours = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == -1:  # Chỉ lấy contours ngoài cùng
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if area > area_threshold and aspect_ratio > aspect_ratio_threshold:
                filtered_contours.append(cnt)

    # Sắp xếp contours từ trái sang phải
    filtered_contours = sorted(filtered_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    digits = []
    rois = []
    areas = []
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Tạo vùng đệm xung quanh chữ số
        m = 0.2  # Tỷ lệ phần đệm
        roi = bin_img[max(0, y - int(m * h)):y + h + int(m * h), max(0, x - int(m * w)):x + w + int(m * w)]

        # Resize ROI về kích thước 28x28
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        # Đảo ngược màu sắc (nền trắng, chữ số đen)
        roi = cv2.bitwise_not(roi)

        digits.append(roi)
        rois.append((x, y, w, h))
        areas.append(area)

    return digits, rois, areas 
