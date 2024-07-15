import base64
import os
import cv2
import json
import numpy as np
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.ocr.v20181119 import ocr_client, models

# 图片路径
image_path = "C:\\Users\\22597\\Documents\\GitHub\\boiling\\C.jpg"

# OCR服务相关参数
SecretId = 
SecretKey = 

# 腾讯云OCR服务
def recognize_digits(image_path):
    try:
        cred = credential.Credential(SecretId, SecretKey)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "ocr.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = ocr_client.OcrClient(cred, "ap-guangzhou", clientProfile)

        # 读取图片
        image = cv2.imread(image_path)
        height, width, channels = image.shape

        # 压缩图片以便于上传到腾讯云
        # resized_image = cv2.resize(image, (int(width / 5), int(height / 5)))
        resized_image = image
        _, compressed_image = cv2.imencode('.jpg', resized_image)

        # 将图片转为Base64编码
        image_base64 = base64.b64encode(compressed_image.tobytes()).decode("utf-8")

        # 调用腾讯云OCR服务
        request = models.GeneralAccurateOCRRequest()
        request.ImageBase64 = image_base64
        response = client.GeneralAccurateOCR(request)

        # 处理响应
        result = response.to_json_string()
        json_result = json.loads(result)
        detections = json_result.get("TextDetections", [])

        # 显示原始图片
        original_image = cv2.imread(image_path)
        output_image = original_image.copy()

        for detection in detections:
            detected_text = detection["DetectedText"]
            x_min, y_min = detection["Polygon"][0]["X"], detection["Polygon"][0]["Y"]
            x_max, y_max = detection["Polygon"][2]["X"], detection["Polygon"][2]["Y"]

            # x_min *= 5
            # y_min *= 5
            # x_max *= 5
            # y_max *= 5

            # 在原图上绘制红色矩形框
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            # 在矩形旁边添加红色文本
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size, baseline = cv2.getTextSize(detected_text, font_face, font_scale, thickness)
            cv2.putText(output_image, detected_text, (x_min, y_max + baseline), font_face, font_scale, (0, 0, 255), thickness)

        # 输出带有标注的图片
        output_path = "tencentOCR_output_C.jpg"
        cv2.imwrite(output_path, output_image)
        print(f"Output file saved to {output_path}")

    except TencentCloudSDKException as err:
        print(err)


if __name__ == "__main__":
    recognize_digits(image_path)