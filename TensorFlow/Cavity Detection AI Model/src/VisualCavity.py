# Original Code for Running Cavity
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array


# 함수: XML 파싱 및 이미지 로드
def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    object_name = root.find('.//object/name').text

    # 수정: 바운딩 박스의 좌표 정보는 XML 파일의 구조에 맞게 수정합니다.
    xmin = float(root.find('.//bndbox/xmin').text)
    ymin = float(root.find('.//bndbox/ymin').text)
    xmax = float(root.find('.//bndbox/xmax').text)
    ymax = float(root.find('.//bndbox/ymax').text)

    return filename, object_name, xmin, ymin, xmax, ymax


# 함수: 이미지 전처리
def preprocess_image(image_path, target_size=(256, 256)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 이미지 정규화
    return img_array


# 데이터 경로 및 XML 파일 경로 설정
image_path = "/content/drive/MyDrive/MyPJT/Cavity_PJT/cavity_sample_img.jpg"  # 실제 이미지 파일의 경로로 수정
xml_path = "/content/drive/MyDrive/MyPJT/Cavity_PJT/cavity_sample_xml.xml"  # XML 파일의 경로로 수정


# XML 파싱 및 데이터 추출
filename, object_name, xmin, ymin, xmax, ymax = parse_annotation(xml_path)


# 이미지 로드 및 전처리
img_array = preprocess_image(image_path)


# 모델 구성
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # 이진 분류 예시


# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 이진 분류 예시
              metrics=['accuracy'])

# 모델 입력 데이터 준비
X_train = np.array([img_array])  # 이미지 데이터를 numpy.ndarray로 변환

# 모델 레이블 데이터 준비
y_train = np.array([1])  # 이진 분류의 경우, 레이블을 numpy.ndarray로 변환

# 이미지 데이터를 모델에 맞는 형태로 변환
X_train = np.squeeze(X_train, axis=0)

# 모델 훈련
model.fit(X_train, y_train, epochs=5, batch_size=1)
