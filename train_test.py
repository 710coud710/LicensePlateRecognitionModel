import numpy as np
import tensorflow  as tf
from sklearn .model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file text chứa mảng 2D
file_path = 'DATADL.txt'
data_2d = np.loadtxt(file_path, dtype=int)

# Tính số lượng mẫu
num_samples = data_2d.size // (28 * 28)  # Số lượng mẫu
data_3d = data_2d.reshape(num_samples, 28, 28)
# In kích thước của mảng 3D
print(data_3d.shape)
print(data_3d)
data_lebel=np.loadtxt("LABELDL.txt",dtype=int)
print(data_lebel.shape)
x_train,x_test,y_train,y_test= train_test_split(data_3d,data_lebel,test_size=0.2,random_state=3)
print(x_train.shape)
print(x_test.shape)
model= tf.keras.models.Sequential()
# Thêm lớp chập đầu tiên với 32 bộ lọc, kích thước kernel (3, 3), padding='same', hàm kích hoạt 'relu', và đầu vào có hình dạng (28, 28, 1)
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
# Thêm lớp chập tiếp theo với 32 bộ lọc, kích thước kernel (3, 3), và hàm kích hoạt 'relu'
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
# Thêm lớp gộp tối đa với kích thước pool là (2, 2)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Thêm lớp dropout với tỷ lệ là 0.25
model.add(tf.keras.layers.Dropout(0.25))
# Thêm lớp chập tiếp theo với 64 bộ lọc, kích thước kernel (3, 3), padding='same', hàm kích hoạt 'relu'
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
# Thêm lớp chập tiếp theo với 64 bộ lọc, kích thước kernel (3, 3), và hàm kích hoạt 'relu'
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# Thêm lớp gộp tối đa với kích thước pool là (2, 2)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Thêm lớp dropout với tỷ lệ là 0.25
model.add(tf.keras.layers.Dropout(0.25))
# Thêm lớp chập tiếp theo với 64 bộ lọc, kích thước kernel (3, 3), padding='same', hàm kích hoạt 'relu'
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
# Thêm lớp chập tiếp theo với 64 bộ lọc, kích thước kernel (3, 3), và hàm kích hoạt 'relu'
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# Thêm lớp gộp tối đa với kích thước pool là (2, 2)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Thêm lớp dropout với tỷ lệ là 0.25
model.add(tf.keras.layers.Dropout(0.25))
# Thêm lớp phẳng hóa dữ liệu
model.add(tf.keras.layers.Flatten())
# Thêm lớp Dense với 512 nút và hàm kích hoạt 'relu'
model.add(tf.keras.layers.Dense(512, activation='relu'))
# Thêm lớp dropout với tỷ lệ là 0.5
model.add(tf.keras.layers.Dropout(0.5))
# Thêm lớp Dense cuối cùng với 32 nút và hàm kích hoạt 'softmax'
model.add(tf.keras.layers.Dense(32, activation='softmax'))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Sử dụng callback để lưu mô hình tốt nhất
checkpoint_path = "nhandien.keras"
model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')
# Hàm callback để in ra số vòng học khi mô hình được lưu
history = model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_test, y_test),callbacks=[model_checkpoint])
#model.save('nhandien300_ok.keras')
# vòng học tốt nhất
best_epoch = np.argmax(history.history['val_accuracy'])  # Tìm vòng học tốt nhất dựa trên val_accuracy
best_loss = history.history['loss'][best_epoch]  # Loss tương ứng với vòng học tốt nhất
best_accuracy = history.history['accuracy'][best_epoch]  # Accuracy tương ứng với vòng học tốt nhất
best_val_loss = history.history['val_loss'][best_epoch]  # Val_loss tương ứng với vòng học tốt nhất
best_val_accuracy = history.history['val_accuracy'][best_epoch]  # Val_accuracy tương ứng với vòng học tốt nhất
# vòng học kém nhất

min_epoch = np.argmin(history.history['val_accuracy'])  # Tìm vòng học tốt nhất dựa trên val_accuracy
min_loss = history.history['loss'][min_epoch]  # Loss tương ứng với vòng học tốt nhất
min_accuracy = history.history['accuracy'][min_epoch]  # Accuracy tương ứng với vòng học tốt nhất
min_val_loss = history.history['val_loss'][min_epoch]  # Val_loss tương ứng với vòng học tốt nhất
min_val_accuracy = history.history['val_accuracy'][min_epoch]  # Val_accuracy tương ứng với vòng học tốt nhất

print(f"Best epoch: {best_epoch + 1}")
print(f"Loss: {best_loss}")
print(f"Accuracy: {best_accuracy}")
print(f"Val_loss: {best_val_loss}")
print(f"Val_accuracy: {best_val_accuracy}")

print("\n\n")
print(f"Min epoch: {min_epoch + 1}")
print(f"Loss: {min_loss}")
print(f"Accuracy: {min_accuracy}")
print(f"Val_loss: {min_val_loss}")
print(f"Val_accuracy: {min_val_accuracy}")

model.summary()
plt.figure(0)
plt.plot(history.history['accuracy'],label='training')
plt.plot(history.history['val_accuracy'],label='val accuracy')
plt.title=('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()



#print(x_train.shape)