import argparse
import cv2
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO object detection and tracking")
    parser.add_argument("--source", default="0", help="Webcam index (0) or path to a video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights; downloaded automatically if absent")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save annotated output as output_tracking.mp4")
    return parser.parse_args()

def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.model)  # First run may download pretrained weights.
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    writer = None
    window_name = "CodeAlpha - Object Detection and Tracking"
    print("Press Q to quit the video window.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # persist=True keeps tracker state between consecutive frames.
            results = model.track(frame, persist=True, conf=args.confidence, tracker="bytetrack.yaml", verbose=False)
            annotated = results[0].plot()

            if args.save and writer is None:
                h, w = annotated.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0 or fps > 120:
                    fps = 25.0
                writer = cv2.VideoWriter("output_tracking.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            if writer is not None:
                writer.write(annotated)
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
