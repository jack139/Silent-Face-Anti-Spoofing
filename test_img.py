# -*- coding: utf-8 -*-

import os, sys
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
#warnings.filterwarnings('ignore')

from insightface.app import FaceAnalysis
from insightface.utils import face_align

app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(224, 224))

model_dir="./resources/anti_spoof_models"
models_list = os.listdir(model_dir)
model_test = AntiSpoofPredict(model_dir)
image_cropper = CropImage()

# 不检测人脸
def fas_check(image_path):
    image = cv2.imread(image_path)

    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    img = cv2.resize(image, (80, 80), interpolation=cv2.INTER_LINEAR)
    for model_name in models_list:
        prediction += model_test.predict(img, model_name)

    #print(prediction)
    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2

    return label==1, value # is live, score



def test(image_path):

    image = cv2.imread(image_path)

    #image_bbox = model_test.get_bbox(image)

    faces = app.get(image, max_num=100) # 检测人脸

    for face in faces:
        left, top, right, bottom = face.bbox
        image_bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]

        prediction = np.zeros((1, 3))
        test_speed = 0
        #img = face_align.norm_crop(image, landmark=face.kps, image_size=80) # 人脸修正

        # sum the prediction from single model's result
        for model_name in models_list:
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            print(image_bbox)
            img = image_cropper.crop(**param)
            #cv2.imwrite('crop_%s.jpg'%model_name, img)

            start = time.time()
            prediction += model_test.predict(img, model_name)
            test_speed += time.time()-start

        #print(prediction)
        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label]/2
        if label == 1:
            print("Image '{}' is Real Face. Score: {:.2f}.".format(image_path, value))
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_path, value))
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))

        # 画框
        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

        format_ = os.path.splitext(image_path)[-1]
        #cv2.imwrite('test'+format_, image)


if __name__ == "__main__":

    if len(sys.argv)<2:
        print("usage: python3 %s <image_path>" % sys.argv[0])
        sys.exit(1)

    img_path = sys.argv[1]

    test(img_path)

    print(fas_check(img_path))
