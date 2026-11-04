
from pathlib import Path
from collections import Counter
import csv
import re

import cv2
import easyocr
from ultralytics import YOLO


# ==================================================
# 1. PROJECT PATHS AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

VIDEO_DIR = PROJECT_DIR / "videos"
MODEL_PATH = PROJECT_DIR / "license_plate_detector.pt"

OUTPUT_CSV = PROJECT_DIR / "plate_readings.csv"

CROPS_DIR = (
    PROJECT_DIR / "plate_crops" / "all_detections"
)

CROPS_DIR.mkdir(parents=True, exist_ok=True)

# Process every 3rd frame
FRAME_SKIP = 3

# Plate detection confidence threshold
DETECTION_CONFIDENCE = 0.25

# OCR confidence threshold for manual review
OCR_REVIEW_THRESHOLD = 0.50

# Set True only if OpenCV GUI works
SHOW_WINDOW = False


# ==================================================
# 2. FIND TRAFFIC VIDEO
# ==================================================

def find_video():

    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Videos folder not found: {VIDEO_DIR}"
        )

    video_files = sorted([
        p for p in VIDEO_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in [
            ".mp4", ".avi", ".mov", ".mkv"
        ]
    ])

    if not video_files:
        raise FileNotFoundError(
            "No traffic video found inside videos folder."
        )

    print("\nAVAILABLE VIDEOS")

    for index, video in enumerate(video_files, start=1):
        print(f"{index}. {video.name}")

    return video_files[0]


# ==================================================
# 3. LOAD MODELS
# ==================================================

def load_models():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"License plate model not found: {MODEL_PATH}"
        )

    print("\nLoading license plate detector...")

    plate_model = YOLO(str(MODEL_PATH))

    print("Loading EasyOCR...")

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    print("Both models loaded successfully!")

    return plate_model, reader


# ==================================================
# 4. IMAGE ENHANCEMENT
# ==================================================

def enhance_plate_image(image):

    if image is None or image.size == 0:
        return None

    # Upscale plate crop
    enlarged = cv2.resize(
        image,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Mild sharpening
    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.5,
        blurred,
        -0.5,
        0
    )

    return sharpened


# ==================================================
# 5. OCR TEXT CLEANING
# ==================================================

def clean_plate_text(text):

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ==================================================
# 6. RECOGNIZE NUMBER PLATE
# ==================================================

def recognize_plate(reader, plate_crop):

    processed = enhance_plate_image(plate_crop)

    if processed is None:
        return "", 0.0

    ocr_results = reader.readtext(
        processed,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        detail=1,
        paragraph=False
    )

    best_text = ""
    best_confidence = 0.0

    for _, text, confidence in ocr_results:

        cleaned = clean_plate_text(text)

        if (
            len(cleaned) >= 4
            and float(confidence) > best_confidence
        ):

            best_text = cleaned
            best_confidence = float(confidence)

    return best_text, best_confidence


# ==================================================
# 7. PROCESS VIDEO
# ==================================================

def process_anpr(video_path):

    plate_model, reader = load_models()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    frame_number = 0
    detection_number = 0

    plate_records = []

    print("\nStarting ANPR processing...")
    print("Press Q to stop if the preview window is enabled.")

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Process selected frames only
            if frame_number % FRAME_SKIP != 0:
                continue

            height, width = frame.shape[:2]

            results = plate_model.predict(
                frame,
                conf=DETECTION_CONFIDENCE,
                verbose=False
            )

            annotated_frame = frame.copy()

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    # Get plate coordinates
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    detector_confidence = float(
                        box.conf[0].item()
                    )

                    # Keep coordinates inside image
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Add margin around plate
                    pad_x = int((x2 - x1) * 0.08)
                    pad_y = int((y2 - y1) * 0.15)

                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)

                    cx2 = min(width, x2 + pad_x)
                    cy2 = min(height, y2 + pad_y)

                    plate_crop = frame[
                        cy1:cy2,
                        cx1:cx2
                    ]

                    if plate_crop.size == 0:
                        continue

                    if (
                        plate_crop.shape[0] < 10
                        or plate_crop.shape[1] < 20
                    ):
                        continue

                    detection_number += 1

                    # Save original crop
                    crop_filename = (
                        f"frame_{frame_number:06d}_"
                        f"plate_{detection_number:04d}.jpg"
                    )

                    crop_path = CROPS_DIR / crop_filename

                    cv2.imwrite(
                        str(crop_path),
                        plate_crop
                    )

                    # OCR
                    plate_text, ocr_confidence = (
                        recognize_plate(
                            reader,
                            plate_crop
                        )
                    )

                    if not plate_text:
                        plate_text = "UNREADABLE"

                    # Review status
                    if plate_text == "UNREADABLE":
                        review_status = "Unreadable"
                    elif ocr_confidence < OCR_REVIEW_THRESHOLD:
                        review_status = "Review required"
                    else:
                        review_status = "Verify visually"

                    timestamp = (
                        (frame_number - 1) / fps
                    )

                    # Save detection information
                    plate_records.append({

                        "plate_text": plate_text,

                        "timestamp_seconds": round(
                            timestamp, 2
                        ),

                        "ocr_confidence": round(
                            ocr_confidence, 3
                        ),

                        "detector_confidence": round(
                            detector_confidence, 3
                        ),

                        "review_status": review_status,

                        "crop_path": str(
                            crop_path.relative_to(PROJECT_DIR)
                        )

                    })

                    # Draw bounding box
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    label = (
                        plate_text
                        if plate_text != "UNREADABLE"
                        else "Plate detected"
                    )

                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    print(
                        f"Frame {frame_number} | "
                        f"Plate: {plate_text} | "
                        f"OCR: {ocr_confidence:.3f} | "
                        f"Detector: {detector_confidence:.3f}"
                    )

            # Optional preview
            if SHOW_WINDOW:

                cv2.imshow(
                    "ANPR - Number Plate Recognition",
                    annotated_frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nProcessing stopped by user.")
                    break

    finally:

        cap.release()

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    return frame_number, plate_records, detection_number


# ==================================================
# 8. SAVE CSV REPORT
# ==================================================

def save_csv(plate_records):

    fieldnames = [
        "plate_text",
        "timestamp_seconds",
        "ocr_confidence",
        "detector_confidence",
        "review_status",
        "crop_path"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(plate_records)

    print("\nCSV report saved:", OUTPUT_CSV)


# ==================================================
# 9. FINAL SUMMARY
# ==================================================

def show_summary(
    frame_number,
    plate_records,
    detection_number
):

    readable_records = [
        record for record in plate_records
        if record["plate_text"] != "UNREADABLE"
    ]

    print("\n" + "=" * 50)
    print("AUTOMATIC NUMBER PLATE RECOGNITION REPORT")
    print("=" * 50)

    print("Frames read:", frame_number)

    print("Plate detections:", detection_number)

    print(
        "Readable OCR candidates:",
        len(readable_records)
    )

    print(
        "Unreadable detections:",
        len(plate_records) - len(readable_records)
    )

    print("\nOCR review summary:")

    review_counts = Counter(
        record["review_status"]
        for record in plate_records
    )

    for status, count in review_counts.items():
        print(f"{status}: {count}")

    print("\nGenerated files:")

    print("CSV:", OUTPUT_CSV)

    print("Plate crops:", CROPS_DIR)

    print("\nNote: OCR readings are candidate text only.")
    print("All registration numbers require visual verification.")

    print("=" * 50)
    print("ANPR PROCESSING COMPLETED")


# ==================================================
# 10. MAIN PROGRAM
# ==================================================

if __name__ == "__main__":

    try:

        video_path = find_video()

        print("\nUsing video:", video_path.name)

        frame_number, plate_records, detection_number = (
            process_anpr(video_path)
        )

        save_csv(plate_records)

        show_summary(
            frame_number,
            plate_records,
            detection_number
        )

    except Exception as error:

        print("\nANPR ERROR:", error)
