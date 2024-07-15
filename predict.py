<<<<<<< HEAD
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # img = cv2.bitwise_not(img)
    
    # plt.imshow(img.squeeze(), cmap='gray')  # Squeeze去除单通道图像的冗余维度
    # plt.title(f"Label: ")
    # plt.show()
    return img

def segment_digits(img):
    digit_sub_imgs = []
    digit_positions = []

    # 使用连通组件方法分割数字
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    MIN_AREA = 100  # 调整阈值以适应不同场景
    valid_labels = [i for i in range(1, labels.max() + 1) if stats[i][4] > MIN_AREA]
    for label in valid_labels:
        x, y, w, h = stats[label][0:4]
        digit_sub_img = img[y:y+h, x:x+w]
        digit_sub_imgs.append(digit_sub_img)
        digit_positions.append((x, y, w, h))  # 记录位置信息
    print(digit_sub_imgs)
    print(digit_positions)

    # digit_sub_imgs = []
    # digit_positions = []
    # # 使用轮廓方法分割数字
    # contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # MIN_AREA = 100  # 同样需要调整阈值
    # valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    # for contour in valid_contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     digit_sub_img = img[y:y+h, x:x+w]
    #     digit_sub_imgs.append(digit_sub_img)
    #     digit_positions.append((x, y, w, h))  # 记录位置信息
    # print(digit_sub_imgs)
    # print(digit_positions)
    # print('--------------------------')
    return digit_sub_imgs, digit_positions

# 3. 加载预训练模型
model = load_model('printed_digit_classifier.h5')

# 4. 识别数字
def recognize_digit(sub_img):
    sub_img = cv2.resize(sub_img, (128, 128))
    sub_img = sub_img.reshape(1, 128, 128, 1) / 255.0  # 标准化并转换为模型所需的输入格式
    prediction = model.predict(sub_img)
    digit_value = np.argmax(prediction[0])
    return digit_value

# 5. 绘制结果
def draw_results(original_img, digit_positions, digit_values):
    for pos, value in zip(digit_positions, digit_values):
        x, y, w, h = pos  # 假设pos为每个数字的左上角坐标及宽高
        cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 画红色方框
        text_pos = (x, y - 10)  # 文本标注位置（假设在方框上方）
        cv2.putText(original_img, str(value), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# 主程序
input_img_path = 'C.jpg'
output_img_path = 'output_image_C.jpg'


original_img = cv2.imread(input_img_path)  # 读取原始图像数据
preprocessed_img = preprocess_image(input_img_path)

digit_sub_imgs, digit_positions = segment_digits(preprocessed_img)

digit_values = []

for sub_img, pos in zip(digit_sub_imgs, digit_positions):
    digit_value = recognize_digit(sub_img)
    digit_values.append(digit_value)


draw_results(original_img, digit_positions, digit_values)

=======
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # img = cv2.bitwise_not(img)
    
    # plt.imshow(img.squeeze(), cmap='gray')  # Squeeze去除单通道图像的冗余维度
    # plt.title(f"Label: ")
    # plt.show()
    return img

def segment_digits(img):
    digit_sub_imgs = []
    digit_positions = []

    # 使用连通组件方法分割数字
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    MIN_AREA = 100  # 调整阈值以适应不同场景
    valid_labels = [i for i in range(1, labels.max() + 1) if stats[i][4] > MIN_AREA]
    for label in valid_labels:
        x, y, w, h = stats[label][0:4]
        digit_sub_img = img[y:y+h, x:x+w]
        digit_sub_imgs.append(digit_sub_img)
        digit_positions.append((x, y, w, h))  # 记录位置信息
    print(digit_sub_imgs)
    print(digit_positions)

    # digit_sub_imgs = []
    # digit_positions = []
    # # 使用轮廓方法分割数字
    # contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # MIN_AREA = 100  # 同样需要调整阈值
    # valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    # for contour in valid_contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     digit_sub_img = img[y:y+h, x:x+w]
    #     digit_sub_imgs.append(digit_sub_img)
    #     digit_positions.append((x, y, w, h))  # 记录位置信息
    # print(digit_sub_imgs)
    # print(digit_positions)
    # print('--------------------------')
    return digit_sub_imgs, digit_positions

# 3. 加载预训练模型
model = load_model('printed_digit_classifier.h5')

# 4. 识别数字
def recognize_digit(sub_img):
    sub_img = cv2.resize(sub_img, (128, 128))
    sub_img = sub_img.reshape(1, 128, 128, 1) / 255.0  # 标准化并转换为模型所需的输入格式
    prediction = model.predict(sub_img)
    digit_value = np.argmax(prediction[0])
    return digit_value

# 5. 绘制结果
def draw_results(original_img, digit_positions, digit_values):
    for pos, value in zip(digit_positions, digit_values):
        x, y, w, h = pos  # 假设pos为每个数字的左上角坐标及宽高
        cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 画红色方框
        text_pos = (x, y - 10)  # 文本标注位置（假设在方框上方）
        cv2.putText(original_img, str(value), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# 主程序
input_img_path = 'C.jpg'
output_img_path = 'output_image_C.jpg'


original_img = cv2.imread(input_img_path)  # 读取原始图像数据
preprocessed_img = preprocess_image(input_img_path)

digit_sub_imgs, digit_positions = segment_digits(preprocessed_img)

digit_values = []

for sub_img, pos in zip(digit_sub_imgs, digit_positions):
    digit_value = recognize_digit(sub_img)
    digit_values.append(digit_value)


draw_results(original_img, digit_positions, digit_values)

>>>>>>> 81c37fa34b06a5f06fc0dd78f7c45157432b6766
cv2.imwrite(output_img_path, original_img)