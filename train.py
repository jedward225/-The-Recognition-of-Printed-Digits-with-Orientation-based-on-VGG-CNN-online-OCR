<<<<<<< HEAD
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

IMG_WIDTH, IMG_HEIGHT = 128, 128  # 图像尺寸
NUM_CLASSES = 10  # 类别数，对应0-9共10个数字
BATCH_SIZE = 64  # 批次大小
EPOCHS = 210  # 训练轮数
VALIDATION_SPLIT = 0.25  # 验证集占比

def load_data(data_dir):
    """
    加载数据集，并将图像转换为灰度，同时标注类别。
    """
    X = []
    y = []

    # 遍历每个数字子目录
    for folder_name in os.listdir(data_dir): 
        folder_path = os.path.join(data_dir, folder_name)
        
        # 读取该数字子目录下的所有图像并转换为灰度
        for img_path in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, img_path), cv2.IMREAD_GRAYSCALE)
            X.append(img)
            y.append(int(folder_name))

    X = np.array(X).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1) / 255.0  # 归一化到[0, 1]区间
    y = np.array(y)
    
    return X, y

def show_image(X, y):
    """
    显示指定数量的训练图像及其对应的类别标签。
    """
    num_images_to_display = 50
    for i in range(30, num_images_to_display):
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.show()

def create_model():
    """
    构建卷积神经网络模型。
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    train_dir = r'C:\\Users\\22597\\Documents\\GitHub\\boiling\\trainpic'

    X, y = load_data(train_dir)
    
    # 划分训练集和验证集，保持类别比例一致
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, stratify=y) #, random_state=12

    model = create_model()

    # 使用ImageDataGenerator增强训练数据
    datagen = ImageDataGenerator(rotation_range=90,  # 旋转角度范围
                                  width_shift_range=1,  # 水平平移范围
                                  height_shift_range=1,  # 垂直平移范围
                                  # zoom_range=0.2,  # 缩放范围
                                  horizontal_flip=True,  # 是否进行水平翻转
                                  fill_mode='nearest')  # 处理边界像素的方式

    # 使用增强数据进行训练
    history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        epochs=EPOCHS,
                        validation_data=(X_val, y_val))
    # 直接使用原始数据进行训练
    # history = model.fit(X_train, y_train,
    #                     batch_size=BATCH_SIZE,
    #                     epochs=EPOCHS,
    #                     validation_data=(X_val, y_val))
    # 保存模型
    model.save('printed_digit_classifier.h5')

    # 可视化训练过程
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.25, 1])
    plt.legend(loc='lower right')
=======
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

IMG_WIDTH, IMG_HEIGHT = 128, 128  # 图像尺寸
NUM_CLASSES = 10  # 类别数，对应0-9共10个数字
BATCH_SIZE = 64  # 批次大小
EPOCHS = 210  # 训练轮数
VALIDATION_SPLIT = 0.25  # 验证集占比

def load_data(data_dir):
    """
    加载数据集，并将图像转换为灰度，同时标注类别。
    """
    X = []
    y = []

    # 遍历每个数字子目录
    for folder_name in os.listdir(data_dir): 
        folder_path = os.path.join(data_dir, folder_name)
        
        # 读取该数字子目录下的所有图像并转换为灰度
        for img_path in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, img_path), cv2.IMREAD_GRAYSCALE)
            X.append(img)
            y.append(int(folder_name))

    X = np.array(X).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1) / 255.0  # 归一化到[0, 1]区间
    y = np.array(y)
    
    return X, y

def show_image(X, y):
    """
    显示指定数量的训练图像及其对应的类别标签。
    """
    num_images_to_display = 50
    for i in range(30, num_images_to_display):
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.show()

def create_model():
    """
    构建卷积神经网络模型。
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    train_dir = r'C:\\Users\\22597\\Documents\\GitHub\\boiling\\trainpic'

    X, y = load_data(train_dir)
    
    # 划分训练集和验证集，保持类别比例一致
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, stratify=y) #, random_state=12

    model = create_model()

    # 使用ImageDataGenerator增强训练数据
    datagen = ImageDataGenerator(rotation_range=90,  # 旋转角度范围
                                  width_shift_range=1,  # 水平平移范围
                                  height_shift_range=1,  # 垂直平移范围
                                  # zoom_range=0.2,  # 缩放范围
                                  horizontal_flip=True,  # 是否进行水平翻转
                                  fill_mode='nearest')  # 处理边界像素的方式

    # 使用增强数据进行训练
    history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        epochs=EPOCHS,
                        validation_data=(X_val, y_val))
    # 直接使用原始数据进行训练
    # history = model.fit(X_train, y_train,
    #                     batch_size=BATCH_SIZE,
    #                     epochs=EPOCHS,
    #                     validation_data=(X_val, y_val))
    # 保存模型
    model.save('printed_digit_classifier.h5')

    # 可视化训练过程
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.25, 1])
    plt.legend(loc='lower right')
>>>>>>> 81c37fa34b06a5f06fc0dd78f7c45157432b6766
    plt.show()