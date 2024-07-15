import streamlit as st
import yolov5
import torch
from typing import List
import cv2
import easyocr
import os
import numpy as np

def inference(
    path2img: str,
    show_img: bool = False,
    size_img: int = 640,
    nms_conf_thresh: float = 0.7,
    max_detect: int = 10,
) -> List[torch.Tensor]:

    model = yolov5.load("keremberke/yolov5m-license-plate")

    model.conf = nms_conf_thresh
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = max_detect

    results = model(path2img, size=size_img)

    if show_img:
        results.show()

    return results.pred

def ocr_easyocr(image: np.ndarray):
    reader = easyocr.Reader(["en"], gpu=False)
    detections = reader.readtext(image)

    plate_no = []
    for line in detections:
        plate_no.append(line[1])

    return " ".join(plate_no)

def run_license_plate_recognition(image_path: str) -> str:
    # Run YOLOv5 inference
    detections = inference(image_path)

    # Assuming the first detection is the license plate
    if detections[0].shape[0] > 0:
        # Extract the bounding box for the first detection
        x1, y1, x2, y2 = map(int, detections[0][0][0:4])
        image = cv2.imread(image_path)
        cropped_plate = image[y1:y2, x1:x2]

        # Run OCR on the cropped plate
        return ocr_easyocr(cropped_plate)

    return "No plate detected"

def app():
    st.header("License Plate Recognition Web App")
    st.subheader("Powered by YOLOv5")
    st.write("Welcome!")

    with st.form("my_uploader"):
        uploaded_file = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
        )
        submit = st.form_submit_button(label="Upload")

    if uploaded_file is not None:
        save_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)  # Ensure the temp directory exists
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if submit:
            text = run_license_plate_recognition(save_path)
            st.write(f"Detected License Plate Number: {text}")

if __name__ == "__main__":
    app()
