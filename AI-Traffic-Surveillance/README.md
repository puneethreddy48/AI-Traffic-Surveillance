# AI-Powered Intelligent Traffic Surveillance and Vehicle Identification System

## Overview
An AI-based traffic surveillance prototype that analyses recorded traffic videos using computer vision.

## Features
- Vehicle detection using YOLO11
- Vehicle tracking using ByteTrack
- Vehicle counting and traffic density estimation
- Number plate detection
- Text extraction using EasyOCR
- Interactive Streamlit dashboard
- CSV-based traffic reports

## Technologies
- Python
- OpenCV
- YOLO11
- ByteTrack
- EasyOCR
- Streamlit
- Pandas
- Plotly

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

Run vehicle analysis:

```bash
python main.py
```

## Project Status
Prototype developed and tested using recorded traffic videos.

## Future Enhancements
- Live CCTV integration
- Improved number plate recognition
- Authorized vehicle database integration
- Enhanced traffic analysis
