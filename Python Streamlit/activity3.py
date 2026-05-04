import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
from collections import Counter

# Cache the model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Video callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLO tracking
    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Object counting per class
    names = results[0].names
    counts = {}

    if results[0].boxes.cls is not None:
        for cls_id in results[0].boxes.cls.tolist():
            name = names[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1

        y = 50

        for name in counts:
            cv2.putText(
                annotated_frame,
                f"{name}: {counts[name]}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            y += 40

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# Start WebRTC streamer
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)