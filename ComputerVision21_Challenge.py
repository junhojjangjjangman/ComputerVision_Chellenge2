# 세그먼테이션, 마스킹, 로컬라이제이션 등

import cv2
import numpy as np
import warnings
import Dir
import matplotlib.pyplot as plt
import os

from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

## 1.1 특성 추출 - 추론을 돕기 위해 사용할 특성 선택
def averagecolor(image):
    return np.mean(image, axis=(0, 1))

def averagecolor_grey(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(grey, axis=(0, 1))

# 챌린지 1: "images/test/58.png"의 평균 색상은?
image_58 = cv2.imread(Dir.dir+"[Dataset] Module 21 images/test/58.png")
cha1 = averagecolor(image_58)
print(cha1)

image_58 = cv2.cvtColor(image_58,cv2.COLOR_BGR2RGB)
plt.imshow(image_58)
plt.title('image_58.png')
#plt.axis('off')
plt.show()

# 챌린지 2: "images/background.png"의 평균 색상은?
background = cv2.imread(Dir.dir+"[Dataset] Module 21 images/background.png")
cha2 = averagecolor(background)
print(cha2)

background = cv2.cvtColor(background,cv2.COLOR_BGR2RGB)
plt.imshow(background)
plt.title('background.png')
#plt.axis('off')
plt.show()

# 챌린지 3: "images/cardgreen_close.png"의 평균 색상은?
cardgreen_close = cv2.imread(Dir.dir+"[Dataset] Module 21 images/cardgreen_close.png")
cha3 = averagecolor(cardgreen_close)
print(cha3)

cardgreen_close = cv2.cvtColor(cardgreen_close,cv2.COLOR_BGR2RGB)
plt.imshow(cardgreen_close)
plt.title('cardgreen_close.png')
#plt.axis('off')
plt.show()

# 챌린지 4: 58.png vs Background. cha1과 cha2 사이의 거리를 계산하십시오.
print(np.linalg.norm(cha2-cha1))
# 챌린지 5: 58.png vs Green. cha1과 cha3 사이의 거리를 계산하십시오.
print(np.linalg.norm(cha3-cha1))

# 챌린지 6: Modify the function averagecolor to convert the image to greyscale before extracting the features
#
#
#
trainX2 = []
trainY2 = []

# 이미지 하위 디렉토리 4개 폴더에 있는 훈련 이미지를 반복합니다.
path = Dir.dir+"[Dataset] Module 21 images/"
for label in ('red', 'green', 'black', 'none'):
    print("Loading training images for the label: " + label)

    # 하위 폴더의 모든 이미지를 읽어옵니다.
    for filename in os.listdir(path + label + "/"):
        img = cv2.imread(path + label + "/" + filename)
        img_features = averagecolor(img)
        trainX2.append(img_features)
        trainY2.append(label)

knnmodel = KNeighborsClassifier(n_neighbors=1)
knnmodel.fit(trainX2, trainY2)

card_localization = cv2.imread(Dir.dir+"[Dataset] Module 21 images/cardred.png")

# 먼저 픽셀을 모델이 허용하는 형식으로 정렬해야합니다 (2D 배열)
# numpy 배열 방법을 사용하면 쉽게 할 수 있습니다.
temp = card_localization.reshape((307200,3))

# 각 픽셀의 색상을 예측
prediction = knnmodel.predict(temp)

# 우리가 익숙한 이미지 모양으로 예측을 다시 정렬 할 수 있습니다.
masklabels = (prediction.reshape((480,640)))

# 관심 있는 클래스에 대한 마스크를 만듭니다! (이전 세션에서 마스크가 무엇인지 생각해보십시오)
canvas = np.zeros(card_localization.shape[:2],dtype="uint8")
canvas[masklabels=="red"]=255

# 가장 큰 영역의 위치를 찾습니다.
(cnts,_) = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=lambda cnts: cv2.boundingRect(cnts)[1])[:1]  # 위에서 아래로 윤곽을 정렬합니다.
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(card_localization, (x,y), (x+w,y+h), (0,255,0),2)     # 경계 상자를 초록색으로 그립니다.

# 결과 표시
cv2.imshow("Localization",canvas)
cv2.imshow("Marked Card",card_localization) #여기에 그려진 경계 상자를 보십시오.
cv2.waitKey(0)
cv2.destroyAllWindows()

card_localization = cv2.imread(Dir.dir+"[Dataset] Module 21 images/cardred.png")

# 먼저 픽셀을 모델이 허용하는 형식으로 정렬해야합니다 (2D 배열)
# numpy 배열 방법을 사용하면 쉽게 할 수 있습니다.
temp = card_localization.reshape((307200,3))

# 각 픽셀의 색상을 예측
prediction = knnmodel.predict(temp)

# 우리가 익숙한 이미지 모양으로 예측을 다시 정렬 할 수 있습니다.
masklabels = (prediction.reshape((480,640)))

# 관심 있는 클래스에 대한 마스크를 만듭니다! (이전 세션에서 마스크가 무엇인지 생각해보십시오)
canvas = np.zeros(card_localization.shape,dtype="uint8")
canvas[masklabels=="green"]=(0,255,0)
canvas[masklabels=="red"]=(0,0,255)
canvas[masklabels=="black"]=(0,0,0)
canvas[masklabels=="none"]=(255,255,255)

# 결과 표시
cv2.imshow("Image Segmentation",canvas)
cv2.imshow("Original Card",card_localization)
#cv2.imshow("Masked Image",cv2.bitwise_and(card_localization,card_localization,mask=canvas))
cv2.waitKey(0)
cv2.destroyAllWindows()

card_localization = cv2.imread(Dir.dir+"[Dataset] Module 21 images/cardmixed.png")
# 빨간거의 비율 구하기
red_total_fraction = np.count_nonzero(canvas)/(480*640)
print(red_total_fraction)
# 먼저 픽셀을 모델이 허용하는 형식으로 정렬해야합니다 (2D 배열)
# numpy 배열 방법을 사용하면 쉽게 할 수 있습니다.
temp = card_localization.reshape((307200,3))

# 각 픽셀의 색상을 예측
prediction = knnmodel.predict(temp)

# 우리가 익숙한 이미지 모양으로 예측을 다시 정렬 할 수 있습니다.
masklabels = (prediction.reshape((480,640)))

# 관심 있는 클래스에 대한 마스크를 만듭니다! (이전 세션에서 마스크가 무엇인지 생각해보십시오)
canvas = np.zeros(card_localization.shape,dtype="uint8")
# canvas[masklabels=="green"]=(0,255,0)
card_localization[masklabels=="red"]=(0,0,255)
# canvas[masklabels=="black"]=(0,0,0)
# canvas[masklabels=="none"]=(255,255,255)
print(canvas[masklabels=="red"].shape)

# 관심 있는 클래스에 대한 마스크를 만듭니다! (이전 세션에서 마스크가 무엇인지 생각해보십시오)
canvas1 = np.zeros(card_localization.shape[:2],dtype="uint8")
canvas1[masklabels=="red"]=255

# 가장 큰 영역의 위치를 찾습니다.
(cnts,_) = cv2.findContours(canvas1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=lambda cnts: cv2.boundingRect(cnts)[1])[:1]  # 위에서 아래로 윤곽을 정렬합니다.
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(card_localization, (x,y), (x+w,y+h), (0,255,0),2)

# 결과 표시

cv2.imshow("Image Segmentation",canvas1)
cv2.imshow("Image Segmentation",card_localization)
#cv2.imshow("Masked Image",cv2.bitwise_and(card_localization,card_localization,mask=canvas))
cv2.waitKey(0)
cv2.destroyAllWindows()