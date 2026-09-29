import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEVICE = "cuda"

YOLO_MODEL = "yolo11n.pt"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

IMAGE_PATH = "bus.jpg"

YOLO_SIZE = 640
DEPTH_SIZE = 518


# ------------------------------------------------------------
# IMAGE
# ------------------------------------------------------------

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

height, width = image_rgb.shape[:2]

print("IMAGE:", width, height)


# ------------------------------------------------------------
# YOLO
# ------------------------------------------------------------

yolo = YOLO(YOLO_MODEL)

result = yolo.predict(
    source=image_bgr,
    imgsz=YOLO_SIZE,
    conf=0.25,
    device=0,
    verbose=False,
)[0]

print()
print("YOLO BOXES:", len(result.boxes))

for i, box in enumerate(result.boxes):

    print(
        i,
        "class =", int(box.cls.item()),
        "conf =", float(box.conf.item()),
        "box =", box.xyxy[0].tolist()
    )


# ------------------------------------------------------------
# DEPTH
# ------------------------------------------------------------

processor = AutoImageProcessor.from_pretrained(
    DEPTH_MODEL
)

depth_model = AutoModelForDepthEstimation.from_pretrained(
    DEPTH_MODEL,
    torch_dtype=torch.float16,
).to(DEVICE).eval()


depth_image = Image.fromarray(
    image_rgb
).resize(
    (DEPTH_SIZE, DEPTH_SIZE)
)

depth_inputs = processor(
    images=depth_image,
    return_tensors="pt",
)

depth_inputs = {
    k: v.to(DEVICE)
    for k, v in depth_inputs.items()
}


# ------------------------------------------------------------
# DEPTH INFERENCE
# ------------------------------------------------------------

with torch.inference_mode():

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):

        outputs = depth_model(
            **depth_inputs
        )

        predicted_depth = outputs.predicted_depth

        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)


print()
print("DEPTH SHAPE:", depth.shape)
print("DEPTH DEVICE:", depth.device)
print("DEPTH DTYPE:", depth.dtype)

print(
    "DEPTH RANGE:",
    float(depth.min()),
    float(depth.max())
)


# ------------------------------------------------------------
# FUSION
# ------------------------------------------------------------

print()
print("=" * 60)
print("FUSION DEBUG")
print("=" * 60)


for i, (box, cls, conf) in enumerate(
    zip(
        result.boxes.xyxy,
        result.boxes.cls,
        result.boxes.conf,
    )
):

    print()
    print("OBJECT", i)

    x1, y1, x2, y2 = box.detach().cpu().tolist()

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    print(
        "original box:",
        x1, y1, x2, y2
    )

    x1 = max(
        0,
        min(width - 1, x1)
    )

    x2 = max(
        0,
        min(width, x2)
    )

    y1 = max(
        0,
        min(height - 1, y1)
    )

    y2 = max(
        0,
        min(height, y2)
    )

    print(
        "clamped box:",
        x1, y1, x2, y2
    )

    if x2 <= x1 or y2 <= y1:

        print("!!! INVALID BOX !!!")

        continue


    # central 50%

    bx1 = x1 + int(
        (x2 - x1) * 0.25
    )

    bx2 = x2 - int(
        (x2 - x1) * 0.25
    )

    by1 = y1 + int(
        (y2 - y1) * 0.25
    )

    by2 = y2 - int(
        (y2 - y1) * 0.25
    )


    print(
        "ROI:",
        bx1, by1, bx2, by2
    )


    roi = depth[
        by1:by2,
        bx1:bx2
    ]


    print(
        "ROI shape:",
        roi.shape
    )

    print(
        "ROI elements:",
        roi.numel()
    )


    if roi.numel() == 0:

        print("!!! EMPTY ROI !!!")

        continue


    depth_value = torch.median(
        roi
    ).item()


    print(
        "DEPTH:",
        depth_value
    )
