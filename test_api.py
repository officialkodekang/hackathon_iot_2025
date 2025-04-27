import requests
import os
import time
import numpy as np
import cv2
from datetime import datetime
import json
from pathlib import Path

# Test parameters
API_URL = "http://localhost:8000"
TEST_DIR = "./test_images"
NUM_IMAGES = 5
FPS = 15

def generate_test_images(output_dir, count):
    """Generate test images with rectangles simulating people"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    image_paths = []
    
    for i in range(count):
        # Create a blank image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some color to simulate a background
        img[:, :, 0] = 50  # Blue channel
        img[:, :, 1] = 50  # Green channel
        img[:, :, 2] = 50  # Red channel
        
        # Add 1-3 rectangles to simulate people
        num_people = np.random.randint(1, 4)
        
        for _ in range(num_people):
            # Generate random position and size for "person"
            height = np.random.randint(100, 250)
            width = int(height / 3)
            x = np.random.randint(0, 640 - width)
            y = np.random.randint(0, 480 - height)
            
            # Draw a "person" (just a rectangle for testing)
            color = (0, 0, 200)  # Red color for person
            cv2.rectangle(img, (x, y), (x + width, y + height), color, -1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        cv2.putText(img, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        # Add image number
        cv2.putText(img, f"Image {i+1}/{count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        # Add person count
        cv2.putText(img, f"People: {num_people}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        # Save the image
        filename = output_dir / f"test_image_{i:05d}.jpg"
        cv2.imwrite(str(filename), img)
        image_paths.append(filename)
        
        print(f"Generated test image {i+1}/{count}: {filename}")
        
        # Short delay to ensure unique timestamps
        time.sleep(0.1)
    
    return image_paths

def upload_images(api_url, image_paths, process_now=True, fps=15):
    """Upload images to the API server"""
    
    # Prepare the API endpoint
    upload_url = f"{api_url}/api/upload/"
    
    # Prepare the form data
    form_data = {
        "process_now": str(process_now).lower(),
        "fps": str(fps)
    }
    
    # Prepare the files
    files = []
    for img_path in image_paths:
        files.append(("files", (os.path.basename(img_path), open(img_path, "rb"), "image/jpeg")))
    
    try:
        # Send request
        print(f"Uploading {len(files)} images to {upload_url}")
        response = requests.post(upload_url, data=form_data, files=files)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        print(f"Upload successful: {result}")
        
        # Get session ID from response
        session_id = result.get("session_id")
        
        # Close all file handles
        for _, (_, file_obj, _) in files:
            file_obj.close()
        
        return result
    
    except Exception as e:
        print(f"Error uploading images: {str(e)}")
        # Close all file handles even in case of error
        for _, (_, file_obj, _) in files:
            file_obj.close()
        
        # Re-raise the exception
        raise

def check_status(api_url, session_id):
    """Check the status of a processing session"""
    
    status_url = f"{api_url}/api/status/{session_id}"
    
    try:
        response = requests.get(status_url)
        response.raise_for_status()
        return response.json()
    
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return None

def download_video(api_url, session_id, output_path=None):
    """Download the processed video"""
    
    download_url = f"{api_url}/api/download/{session_id}"
    
    try:
        # Send request for video
        print(f"Downloading video from {download_url}")
        response = requests.get(download_url, stream=True)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Determine output path if not provided
        if output_path is None:
            output_path = f"test_processed_{session_id}.mp4"
        
        # Save the video
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Video downloaded to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None

def main():
    print("=== Testing Person Detection API ===")
    
    # Step 1: Generate test images
    print("\nGenerating test images...")
    image_paths = generate_test_images(TEST_DIR, NUM_IMAGES)
    
    # Step 2: Upload images to the API
    print("\nUploading images to API...")
    upload_result = upload_images(API_URL, image_paths, process_now=True, fps=FPS)
    
    if not upload_result:
        print("Upload failed. Exiting.")
        return
    
    session_id = upload_result.get("session_id")
    
    if not session_id:
        print("No session ID returned. Exiting.")
        return
    
    print(f"\nSession ID: {session_id}")
    
    # Step 3: Wait for processing to complete
    print("\nWaiting for processing to complete...")
    timeout = 60  # seconds
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = check_status(API_URL, session_id)
        
        if not status:
            print("Failed to get status. Exiting.")
            break
        
        print(f"Processing status: {status.get('status')}")
        
        if status.get("status") == "completed":
            print("\nProcessing complete!")
            print(f"People detected: {status.get('max_people_in_frame', 0)}")
            
            # Step 4: Download the processed video
            output_path = download_video(API_URL, session_id)
            
            if output_path:
                print(f"\nTest successful! Video saved to {output_path}")
            
            return
        
        if status.get("status") == "error":
            print(f"Processing error: {status.get('error')}")
            return
        
        # Wait before checking again
        time.sleep(2)
    
    print("Timeout waiting for processing to complete")

if __name__ == "__main__":
    main() 