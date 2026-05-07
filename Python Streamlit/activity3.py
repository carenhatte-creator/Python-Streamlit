import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from ultralytics import YOLO
import av
import cv2

@st.cache_resource
def load_model():
   
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model(img, conf=0.5, verbose=False, imgsz=320)

    annotated_frame = results[0].plot()

    counts = {}
    if results and results[0].boxes is not None:
        names = results[0].names
        for cls_id in results[0].boxes.cls.tolist():
            name = names[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1

        y_offset = 50
        for name, count in counts.items():
            cv2.putText(
                annotated_frame,
                f"{name}: {count}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            y_offset += 30

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

webrtc_streamer(
    key="object-detection-unique",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    desired_playing_state=True,
)
