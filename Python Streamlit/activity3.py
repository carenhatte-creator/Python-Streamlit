import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from ultralytics import YOLO
import av
import cv2

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

    img = cv2.resize(img, (320, 320))

    results = model(img, conf=0.5, verbose=False)

    annotated_frame = results[0].plot()

    counts = {}

    if results and results[0].boxes is not None:
        names = results[0].names

        for cls_id in results[0].boxes.cls.tolist():
            name = names[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1

        y = 50
        for name, count in counts.items():
            cv2.putText(
                annotated_frame,
                f"{name}: {count}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            y += 30

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# RTC config
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})
webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_send=True,  
)
