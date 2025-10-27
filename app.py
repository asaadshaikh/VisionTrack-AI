import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import supervision as sv
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import os
import time # To calculate FPS
from deep_sort_realtime.deepsort_tracker import DeepSort # Import DeepSORT

# --- Constants ---
CONFIDENCE_THRESHOLD_DEFAULT = 0.3 # Lowered default for tracking
MODEL_PATH = 'yolov8n.pt'
WEBRTC_RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="YOLOv8 Object Detection & Tracking",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title ---
st.title("🤖 YOLOv8 Object Detection & Tracking Engine")
st.write("Upload an image, video, or use your webcam for real-time detection and tracking.")

# --- Model Loading ---
@st.cache_resource # Cache the model to load only once
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        _ = model.model.names # Ensure model names are loaded
        st.success(f"Model '{model_path}' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model '{model_path}': {e}")
        st.stop()

model = load_yolo_model(MODEL_PATH)

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, CONFIDENCE_THRESHOLD_DEFAULT, 0.05
)

# --- Annotator Setup ---
# Use try-except for robustness
try:
    colors = sv.ColorPalette.DEFAULT
    box_annotator = sv.BoxAnnotator(color=colors, thickness=2)
    label_annotator = sv.LabelAnnotator(color=colors, text_color=sv.Color.white(), text_scale=0.5, text_thickness=1)
except AttributeError as e:
     # Check if the error is due to ColorPalette.DEFAULT
     if 'ColorPalette' in str(e):
          st.warning("sv.ColorPalette.DEFAULT not found, trying basic colors as strings.")
          # FIX: Use string names for colors in fallback
          try:
               box_annotator = sv.BoxAnnotator(color="white", thickness=2)
               label_annotator = sv.LabelAnnotator(color="white", text_color="red", text_scale=0.5, text_thickness=1)
               st.info("Using basic 'white'/'red' colors for annotations.")
          except Exception as fallback_e:
               st.error(f"Error initializing annotators even with basic colors: {fallback_e}. Check supervision installation.")
               st.stop()
     # Check if the error is due to sv.Color.white
     elif "'Color' has no attribute 'white'" in str(e):
         st.warning("sv.Color.white() not found, trying basic colors as strings.")
         # FIX: Use string names for colors in primary block and fallback
         try:
             colors = sv.ColorPalette.DEFAULT # Keep attempting default palette first
             box_annotator = sv.BoxAnnotator(color=colors, thickness=2)
             # Use string 'white' for text color if sv.Color.white fails
             label_annotator = sv.LabelAnnotator(color=colors, text_color="white", text_scale=0.5, text_thickness=1)
             st.info("Using default color palette for boxes, 'white' string for text.")
         except AttributeError: # Fallback if ColorPalette.DEFAULT also failed
             st.warning("sv.ColorPalette.DEFAULT also not found, using basic 'white'/'red' strings.")
             box_annotator = sv.BoxAnnotator(color="white", thickness=2)
             label_annotator = sv.LabelAnnotator(color="white", text_color="red", text_scale=0.5, text_thickness=1)
         except Exception as fallback_e:
             st.error(f"Error initializing annotators with string colors: {fallback_e}. Check supervision installation.")
             st.stop()
     else:
          # If the error is different (like missing BoxAnnotator itself), raise it
          st.error(f"Error initializing annotators: {e}. Check supervision installation.")
          st.stop()
except NameError:
    st.error("Supervision library might not be installed correctly.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during annotator setup: {e}")
    st.stop()


# --- Helper Function for Processing Single Image ---
def process_image(image, confidence):
    results = model.predict(image, conf=confidence, verbose=False)
    result = results[0]
    detections = sv.Detections.from_ultralytics(result)

    annotated_image = image.copy()
    labels = []
    if len(detections) > 0 and detections.class_id is not None:
        labels = [
            f"{model.model.names[class_id]} {conf:0.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
            if class_id in model.model.names
        ]
        # Use the globally defined annotators
        try: # Add try-except around annotation calls
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
            if len(labels) == len(detections):
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        except Exception as anno_e:
            st.warning(f"Error during image annotation: {anno_e}")


    return annotated_image, detections

# --- Helper Function for Video/Webcam Frame Processing with Tracking ---
def process_frame_with_tracking(frame, confidence, tracker, frame_counter):
    start_time = time.time() # Start timer for FPS calculation

    # Run YOLO detection
    results = model.predict(frame, conf=confidence, verbose=False)
    result = results[0]
    detections = sv.Detections.from_ultralytics(result)

    # --- DeepSORT Tracking ---
    # 1. Format detections for DeepSORT: [L, T, R, B, Confidence, ClassID]
    formatted_detections = []
    if len(detections) > 0 and detections.xyxy is not None and detections.confidence is not None and detections.class_id is not None:
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            conf = detections.confidence[i]
            cls_id = detections.class_id[i]
            # Ensure bbox has 4 values
            if len(bbox) == 4:
                 formatted_detections.append((bbox, conf, int(cls_id))) # Tuple: (bbox_xyxy, confidence, class_id)

    # 2. Update tracker
    # The tracker expects detections in the format: ( [L,T,W,H], confidence, detection_class )
    deepsort_detections = []
    for bbox_xyxy, conf, cls_id in formatted_detections:
        x1, y1, x2, y2 = bbox_xyxy
        w, h = x2 - x1, y2 - y1
        # Ensure width and height are positive for DeepSORT
        if w > 0 and h > 0:
            deepsort_detections.append(([int(x1), int(y1), int(w), int(h)], conf, cls_id))


    # Update tracker only if there are valid detections for it
    tracks = []
    if deepsort_detections:
         try:
              tracks = tracker.update_tracks(deepsort_detections, frame=frame) # Pass frame for appearance features
         except Exception as track_e:
              # st.warning(f"Error updating tracker: {track_e}") # Optional warning for debugging
              pass # Continue without tracks if update fails


    # 3. Process tracks
    tracked_bboxes_xyxy = []
    track_ids = []
    track_class_ids = []

    for track in tracks:
         # Check if track is confirmed and updated recently
         if not track.is_confirmed() or track.time_since_update > 1:
              continue # Skip tentative or old tracks

         track_id = track.track_id
         # Get bounding box safely
         try:
              ltrb = track.to_ltrb() # Get bounding box in (L, T, R, B) format
              if ltrb is None or len(ltrb) != 4: continue # Skip if invalid bbox
         except:
              continue # Skip if error getting bbox


         class_id = track.get_det_class() # Get the class ID associated by DeepSORT
         # Ensure class_id is valid before adding
         if class_id is not None and class_id in model.model.names:
              # Store for annotation
              tracked_bboxes_xyxy.append(ltrb)
              track_ids.append(track_id)
              track_class_ids.append(class_id)


    # Create Supervision Detections object for tracked objects
    tracked_detections_supervision = sv.Detections.empty() # Start with empty
    if len(tracked_bboxes_xyxy) > 0:
        # Ensure all lists have the same length before creating Detections
        if len(tracked_bboxes_xyxy) == len(track_ids) == len(track_class_ids):
            tracked_detections_supervision = sv.Detections(
                xyxy=np.array(tracked_bboxes_xyxy),
                # Add tracker_id if needed by your specific supervision version or annotators
                # tracker_id=np.array(track_ids).astype(int)
            )
        # else: # Optional: Log mismatch for debugging
            # print(f"Warning: Mismatch in tracked data lengths: boxes={len(tracked_bboxes_xyxy)}, ids={len(track_ids)}, classes={len(track_class_ids)}")


    # --- Annotation ---
    annotated_frame = frame.copy()
    labels = []
    # Only annotate if there are valid tracked detections AND lengths match
    if len(tracked_detections_supervision) > 0 and len(track_ids) == len(tracked_class_ids) == len(tracked_detections_supervision):
        labels = [
            # Include Track ID in the label
            f"#{track_id} {model.model.names[cls_id]}"
            for track_id, cls_id in zip(track_ids, track_class_ids)
            # No need for extra class_id check here as it was done during list population
        ]
        try: # Add try-except around annotation calls
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=tracked_detections_supervision
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=tracked_detections_supervision, labels=labels
            )
        except Exception as anno_e:
            # st.warning(f"Error during frame annotation: {anno_e}") # Optional warning
            pass # Continue without annotation if error occurs


    # --- FPS Calculation ---
    end_time = time.time()
    processing_time = end_time - start_time
    fps = 1 / processing_time if processing_time > 0 else 0
    # Add FPS to the frame
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return annotated_frame


# --- Webcam Video Transformer with Tracking ---
class YOLOv8TrackingTransformer(VideoTransformerBase):
    def __init__(self, model_instance, confidence_level):
        self.model = model_instance
        self.confidence = confidence_level
        # Initialize the tracker here
        self.tracker = DeepSort(max_age=30,  # Max frames to keep track without detection
                                n_init=3,    # Number of consecutive detections to confirm track
                                nms_max_overlap=1.0) # NMS overlap threshold
        self.frame_counter = 0

    def update_confidence(self, new_confidence):
        self.confidence = new_confidence

    def transform(self, frame):
        try:
            img_bgr = frame.to_ndarray(format="bgr24")
            self.frame_counter += 1

            if img_bgr is None or img_bgr.size == 0:
                # print(f"Webcam Transformer: Received empty frame {self.frame_counter}") # Debug
                return np.zeros((100, 100, 3), dtype=np.uint8) # Return black frame

            # Process frame using the helper function
            annotated_frame = process_frame_with_tracking(
                img_bgr, self.confidence, self.tracker, self.frame_counter
            )
            return annotated_frame
        except Exception as e:
            # print(f"Error in Webcam Transformer (frame {self.frame_counter}): {e}") # Debug
            # Attempt to return original frame data if possible
            try:
                return frame.to_ndarray(format="bgr24")
            except:
                return np.zeros((100, 100, 3), dtype=np.uint8) # Fallback black frame


# --- Factory function for Webcam Transformer ---
def create_webcam_transformer():
    # Initialize or retrieve the transformer from session state
    if 'video_transformer_tracking' not in st.session_state:
        # print("Creating NEW transformer instance for webcam") # Debug
        st.session_state.video_transformer_tracking = YOLOv8TrackingTransformer(model, confidence_threshold)
    # else: # Debug
        # print("Reusing existing transformer instance for webcam")

    # Always update the confidence threshold from the slider
    current_confidence = confidence_threshold # Get value from slider at this point
    # Ensure the transformer exists before updating
    if 'video_transformer_tracking' in st.session_state:
        st.session_state.video_transformer_tracking.update_confidence(current_confidence)
    # print(f"Factory updated transformer confidence to {current_confidence}") # Debug

    # Return the instance (will be None if initialization failed somehow, though unlikely here)
    return st.session_state.get('video_transformer_tracking', None)


# --- Main Page Tabs ---
tab1, tab2, tab3 = st.tabs(["🖼️ Image", "🎬 Video", "LIVE Webcam"])

# --- Image Detection Tab ---
with tab1:
    st.header("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

    col1, col2 = st.columns(2) # Create two columns for side-by-side display

    if uploaded_image:
        file_bytes = uploaded_image.read()
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not decode image.")
        else:
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True, channels="BGR")

            if st.button("Detect Objects", key="image_detect_button"):
                with st.spinner("Detecting..."):
                    try:
                        annotated_image, detections = process_image(image, confidence_threshold)
                    except Exception as e:
                         st.error(f"Error during image processing: {e}")
                         annotated_image = image # Show original on error
                         detections = sv.Detections.empty()


                with col2:
                    st.image(annotated_image, caption="Detected Image", use_container_width=True, channels="BGR")

                st.subheader("Detected Objects:")
                if len(detections) > 0 and detections.class_id is not None:
                    detected_items = set()
                    for class_id, confidence in zip(detections.class_id, detections.confidence):
                        if class_id in model.model.names:
                            class_name = model.model.names[class_id]
                            detected_items.add(f"- {class_name} (Confidence: {confidence:.2f})")
                    if detected_items:
                        for item in sorted(list(detected_items)):
                            st.write(item)
                    else:
                        st.write("No objects detected with valid class IDs.")
                else:
                    st.write("No objects detected.")

# --- Video Detection Tab ---
with tab2:
    st.header("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"], key="video_uploader") # Added mkv
    video_path = None
    tfile = None

    if uploaded_video:
        try:
            temp_dir = tempfile.gettempdir()
            file_suffix = os.path.splitext(uploaded_video.name)[1]
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix or '.tmp', dir=temp_dir)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            tfile.close()

            st.video(video_path) # Display original video

            video_output_placeholder = st.empty() # Placeholder for processed frames

            if st.button("Detect & Track Objects in Video", key="video_detect_button"):
                # --- Initialize Tracker for Video ---
                video_tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
                frame_counter = 0
                cap = None
                try:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("Error: Could not open video file.")
                    else:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        progress_bar = st.progress(0, text="Processing video...") if total_frames > 0 else None

                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            if frame is None or frame.size == 0: continue

                            frame_counter += 1
                            annotated_frame = process_frame_with_tracking(
                                frame, confidence_threshold, video_tracker, frame_counter
                            )
                            video_output_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

                            if progress_bar is not None and total_frames > 0:
                                progress_text = f"Processing video... Frame {frame_counter}/{total_frames}"
                                progress_bar.progress(frame_counter / total_frames, text=progress_text)

                        st.success("Video processing complete.")
                        if progress_bar: progress_bar.empty()

                except Exception as e:
                    st.error(f"An error occurred during video processing: {e}")
                finally:
                    if cap is not None and cap.isOpened(): cap.release()
                    # Clean up temporary file
                    if video_path and os.path.exists(video_path) and tempfile.gettempdir() in os.path.abspath(video_path):
                        try: os.remove(video_path)
                        except Exception as cleanup_error: st.warning(f"Could not delete temp video file: {cleanup_error}")

        except Exception as e:
            st.error(f"Error handling uploaded video: {e}")
            if video_path and os.path.exists(video_path) and tempfile.gettempdir() in os.path.abspath(video_path):
                try: os.remove(video_path)
                except Exception as cleanup_error: st.warning(f"Could not delete temp file after error: {cleanup_error}")

# --- Webcam Detection Tab ---
with tab3:
    st.header("Live Webcam Feed with Tracking")
    st.write("Click 'Start' below.")

    # Get selected mode
    selected_mode_string = st.sidebar.radio("WebRTC Mode", ("SENDRECV", "RECVONLY"), index=0, key="webrtc_mode_tracking")
    mode_enum = WebRtcMode.SENDRECV if selected_mode_string == "SENDRECV" else WebRtcMode.RECVONLY

    webrtc_ctx = webrtc_streamer(
        key="yolo-webcam-tracking",
        mode=mode_enum,
        rtc_configuration=WEBRTC_RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=create_webcam_transformer, # Use the factory
        async_processing=True,
    )

    # Display Status and Confidence (check if transformer exists)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Webcam Status")
    if webrtc_ctx.state.playing:
        st.sidebar.write("Status: Running")
        # Display current confidence being used by transformer
        if 'video_transformer_tracking' in st.session_state:
             st.sidebar.write(f"Confidence: {st.session_state.video_transformer_tracking.confidence:.2f}")
    else:
        st.sidebar.write("Status: Stopped")

    st.markdown("""
        **Instructions:**
        1. Allow browser permissions.
        2. Wait for the stream.
        3. Detections with Track IDs will appear.
        4. Adjust confidence using the slider.
        5. The component provides its own 'Stop' button below the feed.
        """)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Model: YOLOv8n | Tracking: DeepSORT")

