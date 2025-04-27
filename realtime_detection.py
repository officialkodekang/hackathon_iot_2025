from ultralytics import YOLO
import cv2
import time
import os

def main():
    # Load the YOLOv8 model
    model_path = './yolov8m.pt'  # Using the medium model for better accuracy
    model = YOLO(model_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20
    
    # Create output video writer
    output_filename = f"realtime_detection_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    # Initialize counters and tracking
    frames_processed = 0
    people_detected = 0
    unique_person_ids = set()
    
    print("Starting real-time detection. Press 'q' to quit.")
    
    try:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image from webcam")
                break
            
            # Process the frame with YOLOv8
            results = model.track(frame, persist=True, conf=0.3, classes=0)  # Class 0 is 'person'
            
            # Get detection information
            if results and len(results) > 0:
                result = results[0]
                
                # Process tracking information
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    # Get track IDs
                    track_ids = result.boxes.id.int().cpu().tolist()
                    for track_id in track_ids:
                        unique_person_ids.add(track_id)
                
                # Draw detection results on the frame
                annotated_frame = result.plot()
                
                # Count people in this frame
                detected_classes = result.boxes.cls.cpu().numpy()
                frame_people_count = sum(1 for cls in detected_classes if int(cls) == 0)  # 0 is the class ID for 'person'
                people_detected = max(people_detected, frame_people_count)
                
                # Display information on the frame
                cv2.putText(
                    annotated_frame, 
                    f"People in frame: {frame_people_count}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # Show the processed frame
                cv2.imshow("Real-time Person Detection", annotated_frame)
                
                # Write the frame to the output video
                output.write(annotated_frame)
                
                frames_processed += 1
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        print(f"Processed {frames_processed} frames")
        print(f"Maximum number of people detected in a single frame: {people_detected}")
        print(f"Total unique people tracked: {len(unique_person_ids)}")
        print(f"Output saved to {output_filename}")
        
        cap.release()
        output.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 