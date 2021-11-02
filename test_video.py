import sys
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
from datetime import datetime

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

from insightface.app import FaceAnalysis
from insightface.utils import face_align

app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(224, 224))

model_dir="./resources/anti_spoof_models"
models_list = os.listdir(model_dir)
model_test = AntiSpoofPredict(model_dir)
image_cropper = CropImage()



if __name__ == '__main__':

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret:
            faces = app.get(frame, max_num=1) # 检测人脸
            
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
                        "org_img": frame,
                        "bbox": image_bbox,
                        "scale": scale,
                        "out_w": w_input,
                        "out_h": h_input,
                        "crop": True,
                    }
                    #print(image_bbox)
                    img = image_cropper.crop(**param)
                    #cv2.imwrite('crop_%s.jpg'%model_name, img)

                    prediction += model_test.predict(img, model_name)

                #print(prediction)
                # draw result of prediction
                label = np.argmax(prediction)
                value = prediction[0][label]/2

                if label == 1:
                    result_text = "RealFace Score: {:.2f}".format(value)
                    color = (255, 0, 0)
                else:
                    result_text = "FakeFace Score: {:.2f}".format(value)
                    color = (0, 0, 255)

                # 画框
                cv2.rectangle(
                    frame,
                    (image_bbox[0], image_bbox[1]),
                    (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                    color, 2)
                cv2.putText(
                    frame,
                    result_text,
                    (image_bbox[0], image_bbox[1] - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

            # Display the resulting frame
            cv2.imshow('To quit press q', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
