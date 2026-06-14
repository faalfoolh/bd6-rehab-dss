# BD6 — Upper Limb Rehabilitation Decision Support System

**Live App:** https://bd6-rehab-dss.streamlit.app

A rehabilitation decision support system that analyses upper limb motion capture data from Xsens IMU sensors to assess movement quality in patients.

## What it does
- Processes sensor data from 4 Xsens MTW2 wireless IMU sensors (hand, wrist, elbow, shoulder)
- Uses machine learning to classify and assess upper limb movements
- Provides decision support for rehabilitation outcomes

## Movements analysed
- Reach & Retrieve
- Cup to Lip
- Wrist Rotation
- Arm Swing

## Tech Stack
- Python, Streamlit
- Xsens MTW2 IMU sensors
- Machine learning pipeline (`bd6_ml_pipeline.ipynb`)
