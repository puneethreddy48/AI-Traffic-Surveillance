
from collections import Counter
from pathlib import Path
import csv

import cv2
from ultralytics import YOLO


# ==================================================
# 1. PROJECT SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR.parent / "videos"
MODEL_PATH = BASE_DIR.parent / "yolo11n.pt"
REPORT_PATH = BASE_DIR / "traffic_report.csv"
OUTPUT_PATH = BASE_DIR / "annotated_traffic.mp4"

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

LOW_TRAFFIC_MAX = 4
MEDIUM_TRAFFIC_MAX = 12

CONFIDENCE = 0.35
FRAME_SKIP = 1

# ==================================================
# 2. TRAFFIC DENSITY CLASSIFICATION
# ==================================================

def classify_density(vehicle_count):
    if vehicle_count <= LOW_TRAFFIC_MAX:
        return "Low"
    elif vehicle_count <= MEDIUM_TRAFFIC_MAX:
        return "Medium"
    return "High"


# ==================================================
# 3. FIND VIDEO
# ==================================================

def find_video():

    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Videos folder not found: {VIDEO_DIR}"
        )

    video_files = [
        file for file in VIDEO_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in [
            ".mp4", ".avi", ".mov", ".mkv"
        ]
    ]

    if not video_files:
        raise FileNotFoundError(
            "No traffic video found inside the videos folder."
        )

    print("\nAVAILABLE VIDEOS")

    for index, file in enumerate(video_files, start=1):
        print(f"{index}. {file.name}")

    return video_files[0]


# ==================================================
# 4. SAVE CSV REPORT
# ==================================================

def save_report(report_rows):

    if not report_rows:
        print("No report data available.")
        return

    fieldnames = list(report_rows[0].keys())

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(report_rows)

    print("\nCSV report saved:", REPORT_PATH)


# ==================================================
# 5. MAIN TRAFFIC ANALYSIS
# ==================================================

def process_traffic():

    print("\n" + "=" * 50)
    print("AI-POWERED INTELLIGENT TRAFFIC SURVEILLANCE")
    print("=" * 50)

    # Check model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    # Find video
    video_path = find_video()

    print("\nSelected video:", video_path.name)

    # Load YOLO
    print("\nLoading YOLO model...")
    model = YOLO(str(MODEL_PATH))
    print("YOLO model loaded successfully.")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Unable to open traffic video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    print("\nVideo processing started.")
    print("Press Q in the video window to stop.")

    # Statistics
    seen_vehicles = set()
    total_by_type = Counter()
    density_counts = Counter()

    frame_number = 0
    processed_frames = 0
    report_rows = []

    # ==================================================
    # 6. PROCESS EACH VIDEO FRAME
    # ==================================================

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print("\nVideo processing completed.")
                break

            frame_number += 1

            # Optional frame skipping
            if (frame_number - 1) % FRAME_SKIP != 0:
                continue

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(VEHICLE_CLASSES.keys()),
                conf=CONFIDENCE,
                verbose=False
            )

            result = results[0]

            current_by_type = Counter()
            annotated_frame = result.plot()

            # Vehicle detection and tracking
            if result.boxes is not None:

                boxes = result.boxes

                for index in range(len(boxes)):

                    class_id = int(boxes.cls[index].item())

                    vehicle_type = VEHICLE_CLASSES.get(class_id)

                    if vehicle_type is None:
                        continue

                    current_by_type[vehicle_type] += 1

                    # Count unique tracked IDs
                    if boxes.id is not None:

                        track_id = int(
                            boxes.id[index].item()
                        )

                        if track_id not in seen_vehicles:

                            seen_vehicles.add(track_id)

                            total_by_type[vehicle_type] += 1

            # ==================================================
            # 7. TRAFFIC DENSITY
            # ==================================================

            visible_count = sum(
                current_by_type.values()
            )

            density = classify_density(visible_count)

            density_counts[density] += 1

            timestamp = (frame_number - 1) / fps

            processed_frames += 1

            report_rows.append({

                "frame": frame_number,

                "time_seconds": round(
                    timestamp, 2
                ),

                "cars_visible": current_by_type["Car"],

                "motorcycles_visible": current_by_type["Motorcycle"],

                "buses_visible": current_by_type["Bus"],

                "trucks_visible": current_by_type["Truck"],

                "total_visible": visible_count,

                "traffic_density": density

            })

            # ==================================================
            # 8. DISPLAY INFORMATION
            # ==================================================

            overlay = [
                "AI TRAFFIC SURVEILLANCE",
                f"Vehicles Visible: {visible_count}",
                f"Cars: {current_by_type['Car']}",
                f"Motorcycles: {current_by_type['Motorcycle']}",
                f"Buses: {current_by_type['Bus']}",
                f"Trucks: {current_by_type['Truck']}",
                f"Traffic Density: {density}",
                f"Unique Track IDs: {len(seen_vehicles)}"
            ]

            y = 30

            for line in overlay:

                cv2.putText(
                    annotated_frame,
                    line,
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                y += 30

            # Display output
            cv2.imshow(
                "AI Intelligent Traffic Surveillance",
                annotated_frame
            )

            # Save annotated video
            writer.write(annotated_frame)

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nProcessing stopped by user.")
                break

    finally:

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # ==================================================
    # 9. SAVE REPORT
    # ==================================================

    save_report(report_rows)

    # ==================================================
    # 10. FINAL SUMMARY
    # ==================================================

    print("\n" + "=" * 50)
    print("FINAL INTELLIGENT TRAFFIC ANALYSIS")
    print("=" * 50)

    print("Frames processed:", processed_frames)

    print("Unique tracked IDs:", len(seen_vehicles))

    print("\nUNIQUE TRACKED IDs BY VEHICLE TYPE")

    for vehicle_type in VEHICLE_CLASSES.values():

        print(
            f"{vehicle_type}: "
            f"{total_by_type[vehicle_type]}"
        )

    print("\nTRAFFIC DENSITY SUMMARY")

    for level in ["Low", "Medium", "High"]:

        print(
            f"{level} density frames: "
            f"{density_counts[level]}"
        )

    print("\nAverage visible vehicles per processed frame:")

    if report_rows:

        average = sum(
            row["total_visible"]
            for row in report_rows
        ) / len(report_rows)

        peak = max(
            row["total_visible"]
            for row in report_rows
        )

        print(round(average, 2))
        print("Peak visible vehicles:", peak)

    print("\nGenerated files:")
    print("CSV report:", REPORT_PATH)
    print("Annotated video:", OUTPUT_PATH)

    print("\nPROGRAM COMPLETED SUCCESSFULLY")


# ==================================================
# 11. RUN PROJECT
# ==================================================

if __name__ == "__main__":

    try:
        process_traffic()

    except Exception as error:

        print("\nERROR:", error)

        cv2.destroyAllWindows()
