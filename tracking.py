from ultralytics import YOLO
import cv2

# Using the pre-trained model instead of a custom trained one
model_path = './yolov8m.pt'
video_path = './sample_video.mp4'
model = YOLO(model_path)

# You need to download a sample video file before running this script
# You can download a sample video with people using:
# curl -L "https://filesamples.com/samples/video/mp4/sample_960x540.mp4" -o "sample_video.mp4"

results = model.track(video_path, persist=True, stream=True, conf=0.25, task='detect')

max_track_id = 0

cap = cv2.VideoCapture(video_path)
output = cv2.VideoWriter("output-video.avi", cv2.VideoWriter_fourcc(*'MPEG'), 25, (int(cap.get(3)),int(cap.get(4))))

for result in results:
    summary = result.summary()
    for s in summary:
      if 'track_id' in s and 'name' in s and s['track_id'] > max_track_id and s['name'] == 'person':
        max_track_id = s['track_id']
    tracked_frame = result.plot()
    output.write(tracked_frame)
    cv2.imshow('frame', tracked_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

output.release()
cap.release()
cv2.destroyAllWindows()
print("Tracking video complete...")
print(f"There are {max_track_id} peoples in video")