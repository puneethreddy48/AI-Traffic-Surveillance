
import os
import re
import time
from pathlib import Path
from collections import defaultdict

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from ultralytics import YOLO


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Traffic Surveillance",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI-Powered Intelligent Traffic Surveillance")
st.caption(
    "Vehicle Detection | Vehicle Tracking | Traffic Density | ANPR"
)


# =====================================================
# 2. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

VIDEO_FOLDER = PROJECT_DIR / "videos"

VEHICLE_MODEL_PATH = PROJECT_DIR / "yolo11n.pt"
PLATE_MODEL_PATH = PROJECT_DIR / "license_plate_detector.pt"

OUTPUT_FOLDER = PROJECT_DIR / "plate_crops" / "all_detections"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TRAFFIC_REPORT = PROJECT_DIR / "traffic_report.csv"
PLATE_REPORT = PROJECT_DIR / "plate_readings.csv"
ANNOTATED_VIDEO = PROJECT_DIR / "annotated_traffic.mp4"


# COCO vehicle class IDs
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# =====================================================
# 3. LOAD AI MODELS
# =====================================================

@st.cache_resource
def load_models():

    vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))
    plate_model = YOLO(str(PLATE_MODEL_PATH))

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    return vehicle_model, plate_model, reader


# =====================================================
# 4. FIND AVAILABLE VIDEOS
# =====================================================

def find_videos():

    extensions = [
        "*.mp4",
        "*.avi",
        "*.mov",
        "*.mkv"
    ]

    videos = []

    for extension in extensions:
        videos.extend(VIDEO_FOLDER.glob(extension))

    return sorted(videos)


# =====================================================
# 5. IMAGE PROCESSING FOR OCR
# =====================================================

def enhance_plate(crop):

    if crop is None or crop.size == 0:
        return None

    crop = cv2.copyMakeBorder(
        crop,
        8, 8, 8, 8,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    crop = cv2.resize(
        crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


def clean_plate_text(text):

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )

    return text


# =====================================================
# 6. SIDEBAR SETTINGS
# =====================================================

with st.sidebar:

    st.header("⚙️ Performance Settings")

    confidence = st.slider(
        "Vehicle confidence",
        0.10,
        0.60,
        0.35,
        0.05
    )

    image_size = st.select_slider(
        "Vehicle detection image size",
        options=[416, 512, 640, 768],
        value=640
    )

    plate_confidence = st.slider(
        "Plate detection confidence",
        0.10,
        0.60,
        0.25,
        0.05
    )

    plate_every = st.select_slider(
        "Plate detection frequency",
        options=[3, 5, 8, 10],
        value=3
    )

    display_every = st.select_slider(
        "Dashboard refresh frequency",
        options=[1, 2, 3, 5],
        value=2
    )

    st.divider()

    st.caption(
        "Higher detection image sizes may improve detection "
        "but require more processing time."
    )


# =====================================================
# 7. VIDEO SELECTION
# =====================================================

video_files = find_videos()

if not video_files:

    st.error(
        "No traffic video found. Add a video to the videos folder."
    )

    st.stop()


selected_video = st.selectbox(
    "🎥 Select Traffic Video",
    video_files,
    format_func=lambda path: path.name
)

st.video(str(selected_video))


# =====================================================
# 8. ANALYSIS BUTTON
# =====================================================

start_button = st.button(
    "▶ Start Complete Traffic Analysis",
    type="primary",
    width="stretch"
)


if start_button:

    # -----------------------------------------------
    # CHECK MODEL FILES
    # -----------------------------------------------

    if not VEHICLE_MODEL_PATH.exists():

        st.error(
            f"Vehicle model missing: {VEHICLE_MODEL_PATH}"
        )

        st.stop()

    if not PLATE_MODEL_PATH.exists():

        st.error(
            f"License plate model missing: {PLATE_MODEL_PATH}"
        )

        st.stop()


    # -----------------------------------------------
    # LOAD MODELS
    # -----------------------------------------------

    with st.spinner("Loading YOLO and EasyOCR models..."):

        try:

            vehicle_model, plate_model, reader = load_models()

        except Exception as error:

            st.error(f"Model loading failed: {error}")
            st.stop()


    # -----------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------

    cap = cv2.VideoCapture(str(selected_video))

    if not cap.isOpened():

        st.error("Unable to open the selected video.")
        st.stop()


    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0


    # -----------------------------------------------
    # DASHBOARD DISPLAY AREAS
    # -----------------------------------------------

    st.divider()
    st.subheader("📹 Live Traffic Analysis")

    video_display = st.empty()
    status_text = st.empty()
    progress_bar = st.progress(0)

    metric_cols = st.columns(5)

    car_metric = metric_cols[0].empty()
    bike_metric = metric_cols[1].empty()
    bus_metric = metric_cols[2].empty()
    truck_metric = metric_cols[3].empty()
    density_metric = metric_cols[4].empty()


    # -----------------------------------------------
    # DATA STORAGE
    # -----------------------------------------------

    traffic_records = []
    plate_records = []

    unique_vehicles = defaultdict(set)

    frame_number = 0
    start_time = time.time()

    video_writer = None
    writer_initialized = False
    analysis_error = None


    # -----------------------------------------------
    # PROCESS VIDEO FRAMES
    # -----------------------------------------------

    try:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            timestamp_seconds = frame_number / fps

            annotated = frame.copy()

            height, width = frame.shape[:2]


            # =======================================
            # VEHICLE DETECTION AND TRACKING
            # =======================================

            results = vehicle_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(VEHICLE_CLASSES.keys()),
                conf=confidence,
                imgsz=image_size,
                iou=0.5,
                max_det=150,
                verbose=False
            )


            counts = {
                "Car": 0,
                "Motorcycle": 0,
                "Bus": 0,
                "Truck": 0
            }


            if results and results[0].boxes is not None:

                for box in results[0].boxes:

                    cls_id = int(box.cls[0].item())

                    if cls_id not in VEHICLE_CLASSES:
                        continue

                    label = VEHICLE_CLASSES[cls_id]

                    counts[label] += 1

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    confidence_value = float(
                        box.conf[0].item()
                    )

                    # Vehicle bounding box
                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        annotated,
                        f"{label} {confidence_value:.2f}",
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

                    # Track ID
                    if box.id is not None:

                        track_id = int(
                            box.id[0].item()
                        )

                        unique_vehicles[label].add(track_id)

                        cv2.putText(
                            annotated,
                            f"ID: {track_id}",
                            (x1, min(y2 + 20, height - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            2
                        )


            # =======================================
            # TRAFFIC DENSITY ANALYSIS
            # =======================================

            visible_total = sum(counts.values())

            if visible_total < 10:

                density = "Low"
                density_color = (0, 200, 0)

            elif visible_total < 25:

                density = "Medium"
                density_color = (0, 200, 255)

            else:

                density = "High"
                density_color = (0, 0, 255)


            overlay_lines = [
                f"Cars: {counts['Car']}",
                f"Bikes: {counts['Motorcycle']}",
                f"Buses: {counts['Bus']}",
                f"Trucks: {counts['Truck']}",
                f"Visible Vehicles: {visible_total}",
                f"Traffic Density: {density}"
            ]


            y_position = 30

            for line in overlay_lines:

                color = (
                    density_color
                    if "density" in line.lower()
                    else (255, 255, 255)
                )

                cv2.putText(
                    annotated,
                    line,
                    (15, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                y_position += 27


            # =======================================
            # NUMBER PLATE DETECTION AND OCR
            # =======================================

            if frame_number % plate_every == 0:

                plate_results = plate_model.predict(
                    frame,
                    conf=plate_confidence,
                    imgsz=640,
                    verbose=False
                )


                if (
                    plate_results
                    and plate_results[0].boxes is not None
                ):

                    for plate_box in plate_results[0].boxes:

                        px1, py1, px2, py2 = map(
                            int,
                            plate_box.xyxy[0].tolist()
                        )

                        px1 = max(0, px1)
                        py1 = max(0, py1)
                        px2 = min(width, px2)
                        py2 = min(height, py2)

                        if px2 <= px1 or py2 <= py1:
                            continue

                        crop = frame[
                            py1:py2,
                            px1:px2
                        ]

                        if crop.size == 0:
                            continue


                        enhanced_crop = enhance_plate(crop)

                        if enhanced_crop is None:
                            continue


                        ocr_results = reader.readtext(
                            enhanced_crop,
                            detail=1,
                            paragraph=False,
                            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                        )


                        if not ocr_results:
                            continue


                        best_result = max(
                            ocr_results,
                            key=lambda item: item[2]
                        )

                        plate_text = clean_plate_text(
                            str(best_result[1])
                        )

                        ocr_confidence = float(
                            best_result[2]
                        )

                        if len(plate_text) < 4:
                            continue


                        # Save plate crop
                        crop_filename = (
                            f"plate_frame_{frame_number}_"
                            f"{len(plate_records) + 1}.jpg"
                        )

                        crop_path = OUTPUT_FOLDER / crop_filename

                        cv2.imwrite(
                            str(crop_path),
                            enhanced_crop
                        )


                        plate_records.append({

                            "frame": frame_number,

                            "timestamp_seconds": round(
                                timestamp_seconds,
                                2
                            ),

                            "plate_text_candidate": plate_text,

                            "ocr_confidence": round(
                                ocr_confidence,
                                4
                            ),

                            "crop_path": str(crop_path)

                        })


                        # Plate bounding box
                        cv2.rectangle(
                            annotated,
                            (px1, py1),
                            (px2, py2),
                            (255, 0, 255),
                            2
                        )

                        cv2.putText(
                            annotated,
                            f"Plate: {plate_text[:18]}",
                            (px1, max(py1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 255),
                            2
                        )


            # =======================================
            # SAVE FRAME DATA
            # =======================================

            traffic_records.append({

                "frame": frame_number,

                "timestamp_seconds": round(
                    timestamp_seconds,
                    2
                ),

                "cars": counts["Car"],

                "motorcycles": counts["Motorcycle"],

                "buses": counts["Bus"],

                "trucks": counts["Truck"],

                "visible_vehicles": visible_total,

                "traffic_density": density

            })


            # =======================================
            # SAVE ANNOTATED VIDEO
            # =======================================

            if not writer_initialized:

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                video_writer = cv2.VideoWriter(
                    str(ANNOTATED_VIDEO),
                    fourcc,
                    fps,
                    (width, height)
                )

                writer_initialized = True

                if not video_writer.isOpened():

                    st.warning(
                        "Could not create annotated video. "
                        "CSV reports will still be generated."
                    )

                    video_writer = None


            if video_writer is not None:

                video_writer.write(annotated)


            # =======================================
            # LIVE DASHBOARD UPDATE
            # =======================================

            if (
                frame_number % display_every == 0
                or frame_number == 1
            ):

                rgb_frame = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                video_display.image(
                    rgb_frame,
                    channels="RGB",
                   width="stretch"
                )

                car_metric.metric(
                    "🚗 Cars",
                    counts["Car"]
                )

                bike_metric.metric(
                    "🏍️ Bikes",
                    counts["Motorcycle"]
                )

                bus_metric.metric(
                    "🚌 Buses",
                    counts["Bus"]
                )

                truck_metric.metric(
                    "🚚 Trucks",
                    counts["Truck"]
                )

                density_metric.metric(
                    "🚦 Density",
                    density
                )


            # =======================================
            # PROGRESS
            # =======================================

            elapsed = max(
                time.time() - start_time,
                0.001
            )

            processing_fps = frame_number / elapsed

            status_text.write(
                f"Processed {frame_number}"
                + (
                    f" / {total_frames}"
                    if total_frames > 0
                    else ""
                )
                + f" frames | Processing speed: {processing_fps:.1f} FPS"
            )

            if total_frames > 0:

                progress_bar.progress(
                    min(
                        frame_number / total_frames,
                        1.0
                    )
                )


    except Exception as error:

        analysis_error = error

        st.error(
            f"Analysis stopped because of an error: {error}"
        )


    finally:

        cap.release()

        if video_writer is not None:
            video_writer.release()


    # =================================================
    # 9. SAVE CSV REPORTS
    # =================================================

    traffic_df = pd.DataFrame(traffic_records)

    plates_df = pd.DataFrame(plate_records)

    traffic_df.to_csv(
        TRAFFIC_REPORT,
        index=False
    )

    plates_df.to_csv(
        PLATE_REPORT,
        index=False
    )


    if traffic_df.empty:

        st.error("No video frames were processed.")

        st.stop()


    if analysis_error is None:

        st.success(
            "Traffic analysis completed successfully!"
        )

    else:

        st.warning(
            "Partial analysis results were saved."
        )


    # =================================================
    # 10. TRAFFIC SUMMARY
    # =================================================

    st.divider()

    st.header("📊 Traffic Analysis Summary")

    total_unique = sum(
        len(ids)
        for ids in unique_vehicles.values()
    )

    peak_vehicles = int(
        traffic_df["visible_vehicles"].max()
    )

    high_density_frames = int(
        (
            traffic_df["traffic_density"] == "High"
        ).sum()
    )


    summary_cols = st.columns(5)

    summary_cols[0].metric(
        "Total Frames",
        len(traffic_df)
    )

    summary_cols[1].metric(
        "Unique Track IDs",
        total_unique
    )

    summary_cols[2].metric(
        "Peak Vehicles",
        peak_vehicles
    )

    summary_cols[3].metric(
        "High Density Frames",
        high_density_frames
    )

    summary_cols[4].metric(
        "Plate OCR Candidates",
        len(plates_df)
    )


    st.caption(
        "Unique Track IDs are tracker-generated IDs, not guaranteed "
        "unique real-world vehicle identities. Density is based on "
        "visible vehicle counts per frame."
    )


    # =================================================
    # 11. VEHICLE CATEGORY SUMMARY
    # =================================================

    st.subheader("🚘 Vehicle Category Summary")

    vehicle_summary = pd.DataFrame({

        "Vehicle Type": [
            "Car",
            "Motorcycle",
            "Bus",
            "Truck"
        ],

        "Unique Track IDs": [
            len(unique_vehicles["Car"]),
            len(unique_vehicles["Motorcycle"]),
            len(unique_vehicles["Bus"]),
            len(unique_vehicles["Truck"])
        ]

    })

    col1, col2 = st.columns(2)

    with col1:

        st.dataframe(
            vehicle_summary,
            width="stretch",
            hide_index=True
        )

    with col2:

        fig = px.bar(
            vehicle_summary,
            x="Vehicle Type",
            y="Unique Track IDs",
            title="Vehicle Tracking Summary",
            text="Unique Track IDs"
        )

        st.plotly_chart(
            fig,
           width="stretch"

        )


    # =================================================
    # 12. TRAFFIC DENSITY GRAPH
    # =================================================

    st.subheader("🚦 Traffic Density Analysis")

    density_summary = (
        traffic_df["traffic_density"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High"],
            fill_value=0
        )
        .reset_index()
    )

    density_summary.columns = [
        "Traffic Density",
        "Number of Frames"
    ]

    fig_density = px.pie(
        density_summary,
        names="Traffic Density",
        values="Number of Frames",
        title="Traffic Density Distribution",
        hole=0.4
    )

    st.plotly_chart(
        fig_density,
        width="stretch"
    )


    # =================================================
    # 13. VEHICLE COUNT GRAPH
    # =================================================

    st.subheader("📈 Vehicle Count Over Time")

    fig_count = px.line(
        traffic_df,
        x="timestamp_seconds",
        y="visible_vehicles",
        title="Visible Vehicle Count Per Frame",
        labels={
            "timestamp_seconds": "Time (seconds)",
            "visible_vehicles": "Visible Vehicles"
        }
    )

    st.plotly_chart(
        fig_count,
        width="stretch"
    )


    # =================================================
    # 14. NUMBER PLATE RESULTS
    # =================================================

    st.divider()

    st.header("🔎 Number Plate Recognition Results")

    if not plates_df.empty:

        st.warning(
            "OCR results are candidate readings, not verified "
            "registration numbers. Check the saved plate crops "
            "before relying on any reading."
        )

        st.metric(
            "Plate OCR Candidates",
            len(plates_df)
        )

        st.dataframe(
            plates_df,
            width="stretch",
            hide_index=True
        )

        st.subheader("Saved Plate Crops")

        for index, row in plates_df.head(6).iterrows():

            crop_path = Path(row["crop_path"])

            if crop_path.exists():

                st.image(
                    str(crop_path),
                    caption=(
                        f"Candidate: {row['plate_text_candidate']} | "
                        f"OCR confidence: {row['ocr_confidence']}"
                    ),
                    width=300
                )

    else:

        st.info(
            "No readable plate text was detected. "
            "Clearer footage may improve OCR results."
        )


    # =================================================
    # 15. ANNOTATED VIDEO OUTPUT
    # =================================================

    st.divider()

    st.header("🎬 Processed Traffic Video")

    if ANNOTATED_VIDEO.exists():

        st.video(str(ANNOTATED_VIDEO))

        st.download_button(
            "⬇️ Download Annotated Video",
            data=ANNOTATED_VIDEO.read_bytes(),
            file_name="annotated_traffic.mp4",
            mime="video/mp4"
        )


    # =================================================
    # 16. DOWNLOAD REPORTS
    # =================================================

    st.divider()

    st.header("📁 Download Analysis Reports")

    download_cols = st.columns(2)

    with download_cols[0]:

        st.download_button(
            "⬇️ Download Traffic Report",
            data=TRAFFIC_REPORT.read_bytes(),
            file_name="traffic_report.csv",
            mime="text/csv",
            width="stretch"
        )

    with download_cols[1]:

        st.download_button(
            "⬇️ Download ANPR Report",
            data=PLATE_REPORT.read_bytes(),
            file_name="plate_readings.csv",
            mime="text/csv",
            width="stretch"
        )


    st.success(
        "All available analysis results and reports have been saved."
    )


# =====================================================
# 17. FOOTER
# =====================================================

st.divider()

st.caption(
    "AI-Powered Intelligent Traffic Surveillance and "
    "Vehicle Identification System"
)
