from ultralytics import YOLO
import cv2

# Using the pre-trained model
model_path = './yolov8m.pt'
video_path = './sample_video.mp4'
model = YOLO(model_path)

# Run detection on the video file
results = model(video_path, stream=True)

# Process the results
for i, result in enumerate(results):
    # Get the classes detected in this frame
    classes = result.boxes.cls.cpu().numpy()
    names = [model.names[int(cls)] for cls in classes]
    
    # Print the classes detected in this frame
    print(f"Frame {i}: Detected {len(classes)} objects: {names}")
    
    # Draw the results on the frame
    annotated_frame = result.plot()
    
    # Display the annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
print("Detection complete!") 