import cv2
import numpy as np

INPUT = "testvideodepth.mp4"
OUTPUT = "testvideodepth_stabilized.mp4"

cap = cv2.VideoCapture(INPUT)

if not cap.isOpened():
    raise RuntimeError(f"Could not open {INPUT}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input: {width}x{height} @ {fps:.3f} FPS")
print(f"Frames: {n_frames}")

# ------------------------------------------------------------
# 1. Estimate frame-to-frame camera motion
# ------------------------------------------------------------

transforms = []

ret, prev = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

for i in range(n_frames - 1):

    ret, curr = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Detect good features in previous frame
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7
    )

    if prev_pts is None:
        transforms.append([0, 0, 0])
        prev_gray = curr_gray
        continue

    # Track features into current frame
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts,
        None
    )

    if curr_pts is None:
        transforms.append([0, 0, 0])
        prev_gray = curr_gray
        continue

    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]

    if len(good_prev) < 10:
        transforms.append([0, 0, 0])
        prev_gray = curr_gray
        continue

    # Estimate rigid camera motion:
    # rotation + translation
    matrix, _ = cv2.estimateAffinePartial2D(
        good_prev,
        good_curr,
        method=cv2.RANSAC,
        ransacReprojThreshold=3
    )

    if matrix is None:
        transforms.append([0, 0, 0])
    else:
        dx = matrix[0, 2]
        dy = matrix[1, 2]
        da = np.arctan2(matrix[1, 0], matrix[0, 0])

        transforms.append([dx, dy, da])

    prev_gray = curr_gray

transforms = np.asarray(transforms)

print("Camera motion estimated.")

# ------------------------------------------------------------
# 2. Build cumulative camera trajectory
# ------------------------------------------------------------

trajectory = np.cumsum(transforms, axis=0)

# ------------------------------------------------------------
# 3. Smooth trajectory
# ------------------------------------------------------------

def smooth(x, radius=30):
    window = 2 * radius + 1

    kernel = np.ones(window) / window

    padded = np.pad(
        x,
        (radius, radius),
        mode="edge"
    )

    smoothed = np.convolve(
        padded,
        kernel,
        mode="same"
    )

    return smoothed[radius:-radius]


smoothed = np.zeros_like(trajectory)

for i in range(3):
    smoothed[:, i] = smooth(
        trajectory[:, i],
        radius=30
    )

# Difference between desired smooth trajectory
# and measured trajectory
difference = smoothed - trajectory

# ------------------------------------------------------------
# 4. Second pass: stabilize frames
# ------------------------------------------------------------

cap.release()
cap = cv2.VideoCapture(INPUT)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT,
    fourcc,
    fps,
    (width, height)
)

ret, frame = cap.read()

for i in range(len(transforms)):

    if not ret:
        break

    dx, dy, da = transforms[i]

    # Apply correction based on trajectory difference
    correction = difference[i]

    transform = np.array([
        [
            np.cos(da),
            -np.sin(da),
            dx + correction[0]
        ],
        [
            np.sin(da),
            np.cos(da),
            dy + correction[1]
        ]
    ], dtype=np.float32)

    stabilized = cv2.warpAffine(
        frame,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Crop slightly to hide edge artifacts
    crop = 0.05

    x1 = int(width * crop)
    x2 = int(width * (1 - crop))
    y1 = int(height * crop)
    y2 = int(height * (1 - crop))

    cropped = stabilized[y1:y2, x1:x2]

    output_frame = cv2.resize(
        cropped,
        (width, height)
    )

    writer.write(output_frame)

    if i % 100 == 0:
        print(f"Processed {i}/{len(transforms)} frames")

    ret, frame = cap.read()

cap.release()
writer.release()

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print(f"Saved: {OUTPUT}")
