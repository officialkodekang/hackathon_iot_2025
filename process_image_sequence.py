from ultralytics import YOLO
import cv2
import os
import glob
import time
import numpy as np
from pathlib import Path

def process_image_sequence(image_folder, output_path=None, image_extension='jpg', fps=15):
    """
    Process a sequence of images from a folder, detect people, and create a video.
    
    Args:
        image_folder: Path to folder containing image sequence
        output_path: Path to save output video (default: auto-generated)
        image_extension: Extension of image files to process (default: jpg)
        fps: Frames per second for output video (default: 15)
    """
    # Check if folder exists
    if not os.path.isdir(image_folder):
        print(f"Error: Folder '{image_folder}' does not exist")
        return
    
    # Get all images in the folder
    image_pattern = os.path.join(image_folder, f"*.{image_extension}")
    image_paths = sorted(glob.glob(image_pattern))
    
    if not image_paths:
        print(f"Error: No {image_extension} images found in '{image_folder}'")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Generate output path if not provided
    if output_path is None:
        output_path = f"processed_sequence_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    
    # Load the YOLOv8 model
    model_path = './yolov8m.pt'
    model = YOLO(model_path)
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"Error: Could not read image {image_paths[0]}")
        return
    
    height, width = first_image.shape[:2]
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each image
    total_people = 0
    max_people_in_frame = 0
    processed_count = 0
    
    for img_path in image_paths:
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}, skipping")
            continue
        
        # Process with YOLOv8
        results = model.track(image, persist=True, conf=0.3, classes=0)  # Class 0 is 'person'
        
        if results and len(results) > 0:
            result = results[0]
            
            # Draw detection results on the image
            annotated_frame = result.plot()
            
            # Count people in this frame
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                detected_classes = result.boxes.cls.cpu().numpy()
                people_count = sum(1 for cls in detected_classes if int(cls) == 0)
                max_people_in_frame = max(max_people_in_frame, people_count)
                total_people += people_count
                
                # Add text to the frame
                cv2.putText(
                    annotated_frame, 
                    f"People detected: {people_count}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # Get image filename
                filename = Path(img_path).name
                cv2.putText(
                    annotated_frame, 
                    f"Image: {filename}", 
                    (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
            
            # Write frame to video
            video_writer.write(annotated_frame)
            
            # Show progress
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(image_paths)} images")
                
            # Optionally display each processed frame (uncomment if needed)
            # cv2.imshow("Processing Image Sequence", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    
    # Release resources
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete:")
    print(f"- Processed {processed_count} images")
    print(f"- Maximum people detected in a single frame: {max_people_in_frame}")
    print(f"- Total person detections: {total_people}")
    print(f"- Output video saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process a sequence of images to detect people")
    parser.add_argument("image_folder", help="Folder containing image sequence")
    parser.add_argument("--output", "-o", help="Output video file path")
    parser.add_argument("--ext", "-e", default="jpg", help="Image file extension (default: jpg)")
    parser.add_argument("--fps", "-f", type=int, default=15, help="Frames per second for output video (default: 15)")
    
    args = parser.parse_args()
    
    process_image_sequence(args.image_folder, args.output, args.ext, args.fps) 