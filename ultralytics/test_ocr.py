import torch  # Preloads structural CUDA context links
import onnxruntime as ort
import os
from paddleocr import PaddleOCR
from rapidocr_onnxruntime import RapidOCR

# Enforce dynamic library loading matching your execution provider profile
try:
    ort.preload_dlls(cuda=True, cudnn=True)
except AttributeError:
    pass  # Preload handling varies down inside minor wheel releases

files = [
    '/home/shatviksmit/VisionAid/ultralytics/coco100/val2017/000000187734.jpg',
    '/home/shatviksmit/VisionAid/ultralytics/coco100/val2017/000000088432.jpg'
]

# Initialize Models
p = PaddleOCR(
    text_detection_model_name='PP-OCRv6_tiny_det',
    text_recognition_model_name='PP-OCRv6_tiny_rec',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_limit_side_len=640,
    text_det_limit_type='max',
    text_det_thresh=0.2,
    text_det_box_thresh=0.3,
    engine='onnxruntime',
    device='gpu:0'
)
r = RapidOCR()

# Inference Loop
for f in files:
    print(f"\n==================== {os.path.basename(f)} ====================")
    
    # PaddleOCR Execution
    print("\nPP-OCRv6 Tiny:")
    try:
        pr = p.predict(f)
        if pr and pr[0]:
            # Correct structural mapping for text detection/recognition arrays
            formatted_results = []
            for box_info in pr[0]:
                text = box_info[1][0]
                score = round(float(box_info[1][1]), 3)
                formatted_results.append((text, score))
            print(formatted_results)
        else:
            print("NONE")
    except Exception as e:
        print(f"PaddleOCR Error: {e}")

    # RapidOCR Execution
    print("\nRapidOCR:")
    try:
        rr = r(f)
        if rr and rr[0]:
            print([(line[1], round(float(line[2]), 3)) for line in rr[0]])
        else:
            print("NONE")
    except Exception as e:
        print(f"RapidOCR Error: {e}")
