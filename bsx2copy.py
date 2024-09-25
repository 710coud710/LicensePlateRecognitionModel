# cac thu vien can thiet
import numpy as np
from imutils import perspective
from ultralytics import YOLO
import cv2
import tensorflow  as tf
from tensorflow.keras.models import load_model
model = YOLO("runs/detect/train2/weights/best.pt")
# chọn ảnh cần phát hiện
# tùy chọn save= True nếu muốn lưu lại ảnh , ảnh được lưu vào runs/detect/predict
# bỏ nếu không có nhu cầu
results =model("22a.jpg")
#print(results)
for result in results :
    boxes=result.cpu().boxes.numpy()
    for box in boxes :
        box = boxes.xyxy
x_min = box[0,0]
y_min = box[0,1]
x_max = box[0,2]
y_max = box[0,3]

# Đọc ảnh gốc
image_path = "22a.jpg"
image = cv2.imread(image_path)

# Chuyển đổi tọa độ thành số nguyên
x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

# Cắt vùng biển số trên ảnh gốc theo hộp giới hạn
image = image[y_min:y_max, x_min:x_max]

#lấy chiều dài và rộng của ảnh vùng biển số
height, width = image.shape[:2]
# Tăng kích thước ảnh lên 0.5 lần so với kích thước ban đầu
new_height = int(height * 4)
new_width = int(width * 4)

# Sử dụng hàm resize để thay đổi kích thước
image = cv2.resize(image, (new_width, new_height))

# Hiển thị ảnh cắt được va da tang kich thuoc (tùy chọn)
#cv2.imshow("Cropped Image", image)
#cv2.waitKey(0)
# thong số cho điều kiện tìm counter hợp lệ
width_kytu = 28
height_kytu = 28
min = 0.01
max = 0.09

# Cắt bớt các phần viền của ảnh biển số
border_size = 2  # Thay đổi giá trị này theo nhu cầu
height, width = image.shape[:2]
cropped_image = image[border_size:height - border_size, border_size:width - border_size]
gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("anh xam",gray)
# Áp dụng ngưỡng để tạo ảnh nhị phân
_, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# binary_image=cv2.resize(binary_image,(0,0),fx=2,fy=2)
# cv2.imshow("Anh goc", image)
#cv2.imshow("Anh sau khi nhi phan va cat bot vien", binary_image)
#cv2.waitKey(0)
# Tạo danh sách để lưu đường viền
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# print(contours)
# Vẽ đường viền lên ảnh gốc
result = cropped_image.copy()
# cv2.imshow('anh sau khi cat',result)
# Vẽ tất cả các đường viền
cv2.drawContours(result, contours, -1, (100, 40, 255),3)  # Vẽ tất cả đường viền (tham số thứ tư là màu trắng, tham số thứ năm là độ dày)
# Hiển thị ảnh với đường viền
# cv2.imshow("Anh ve tat ca contours", result)
# cv2.waitKey(0)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
# cv2.imshow("Anh voi toan bo contours", result)
# cv2.waitKey(0)
# Hiển thị ảnh với đường viền và hình chữ nhật
#cv2.imshow("Anh voi contours va hinh chu nhat", result)
#cv2.waitKey(0)

# tim cac contours hop le
char_x_ind = {}
char_x = []
height, width, _ = cropped_image.shape
dientichanh = height * width  # tinh dien tich cua anh
for ind, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour)
    tilerongvacao = w / h
    dientichkytu = w * h
    #area = cv2.contourArea(contour)
   # print(f"Contour {ind + 1} - Area: {area}")
    #  print(dientichkytu)
    if (min * dientichanh < dientichkytu < max * dientichanh) and (0.2 < tilerongvacao < 0.9):
        if x in char_x:  # Nếu vị trí x đã tồn tại, thì tăng x lên để đảm bảo không ghi đè lên ký tự khác
            x = x + 1
        char_x.append(x)  # Thêm vị trí x vào danh sách
        char_x_ind[x] = ind  # Lưu chỉ số của ký tự theo vị trí x
# vẽ các contour hop le
#hien thi cac contour hop le
image_copy=cropped_image.copy()
for i in char_x:
        char_x_value = i  # Lấy giá trị x của từng ký tự
        (x, y, w, h) = cv2.boundingRect(contours[char_x_ind[i]])
        #print("tl=",w/h)
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255,0), 2)
cv2.imshow("Anh co contour hop le ",image_copy)
cv2.waitKey(0)
luukytu = ""
luukytu1=""
char_x = sorted(char_x)
luukytu = ""
luukytu1=""
# Bang dung de anh xa du doan cua 32 lop
KYTU = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}


model = load_model("nhandien.keras")
for i in char_x:
    char_x_value = i  # Lấy giá trị x của từng ký tự
    (x, y, w, h) = cv2.boundingRect(contours[char_x_ind[i]])
    cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    tachvungkytu = gray[y:y + h, x:x + w]  # cat anh chua vung ky tu
    doikichthuocvungkytu = cv2.resize(tachvungkytu, (width_kytu, height_kytu))  # resize anh
    # Đảm bảo rằng ảnh đầu vào có 1 kênh (nếu là ảnh xám)
  #  cv2.imshow("gvdeg", doikichthuocvungkytu)
  #  cv2.waitKey(0)
  #  doikichthuocvungkytu = np.invert(=uocvungkytu))
    doikichthuocvungkytu = doikichthuocvungkytu.reshape(1,28, 28, 1)  # (batch_size, height, width, channels)
   # cv2.imshow("defd", doikichthuocvungkytu2)
   # cv2.waitKey(0)
    print(doikichthuocvungkytu.shape)
    prediction=model.predict(doikichthuocvungkytu)
    predicted_class = np.argmax(prediction)
    print(predicted_class)
    kytu= KYTU[predicted_class]
    if (y < height / 4):
        luukytu = luukytu + kytu
    else:
        luukytu1 = luukytu1 + kytu
        # viet chữ lên ký tự
    cv2.putText(cropped_image, kytu, (x, y + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1)
cv2.imshow("anh nhan dien ", image)
# Chờ người dùng nhấn phím bất kỳ để thoát
cv2.waitKey(0)
cv2.destroyAllWindows()
# ghi ra text
# hien thi khi bien co hai dòng
if (len(luukytu) != 0 and len(luukytu1) != 0):
    print("Bien so nhan dang duoc :", luukytu, "-", luukytu1)
# truong hop bien mot dong
else:
    bs = luukytu[:3] + "-" + luukytu[3:]
    print("Bien so xe nhan dang duoc :", bs)

#print(prediction)
    #chuyển về ký tự