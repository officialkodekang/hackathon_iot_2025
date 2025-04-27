import os
import time
import requests
import json
import RPi.GPIO as GPIO
from pathlib import Path
import argparse

# ----- Configuration -----
FSR_PIN = 17  # GPIO pin connected to FSR
CAPTURE_INTERVAL = 0.2  # Seconds between each photo
CAPTURE_COUNT = 15  # Number of images to capture when triggered
TEMP_DIR = "/tmp/person_detection"
API_URL = "http://172.20.10.10:8000"  # Change to your server IP

# ----- Initialize GPIO -----
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(FSR_PIN, GPIO.IN)
    print("GPIO initialized for FSR on pin", FSR_PIN)

# ----- Photo Capture Function -----
def capture_image(file_path):
    """Capture a photo using raspistill command"""
    cmd = f"raspistill -o {file_path} -w 640 -h 480 -q 70 -n -t 250"
    os.system(cmd)
    return os.path.exists(file_path)

def capture_image_sequence(output_dir, count, interval):
    """Capture a sequence of images"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    image_paths = []
    
    print(f"Capturing {count} images with {interval}s interval")
    for i in range(count):
        # Generate filename with timestamp
        timestamp = int(time.time() * 1000)
        filename = output_dir / f"image_{i:05d}_{timestamp}.jpg"
        
        # Capture image
        if capture_image(str(filename)):
            image_paths.append(filename)
            print(f"Captured image {i+1}/{count}")
        else:
            print(f"Failed to capture image {i+1}/{count}")
        
        # Wait for next capture (except after the last image)
        if i < count - 1:
            time.sleep(interval)
    
    return image_paths

# ----- API Communication Functions -----
def upload_images(api_url, image_paths, process_now=True, fps=15):
    """Upload images to the API server"""
    upload_url = f"{api_url}/api/upload/"
    
    form_data = {
        "process_now": str(process_now).lower(),
        "fps": str(fps)
    }
    
    files = []
    try:
        for img_path in image_paths:
            files.append(("files", (os.path.basename(img_path), open(img_path, "rb"), "image/jpeg")))
        
        print(f"Uploading {len(files)} images to {api_url}")
        response = requests.post(upload_url, data=form_data, files=files)
        response.raise_for_status()
        
        result = response.json()
        print(f"Upload successful: {result}")
        
        # Close all file handles
        for _, (_, file_obj, _) in files:
            file_obj.close()
        
        return result
    
    except Exception as e:
        print(f"Error uploading images: {str(e)}")
        # Close any open file handles
        for _, (_, file_obj, _) in files:
            file_obj.close()
        return None

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
        print(f"Downloading video from {download_url}")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        if output_path is None:
            output_path = f"processed_{session_id}.mp4"
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Video downloaded to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None

def wait_for_processing(api_url, session_id, timeout=300, download=True):
    """Wait for processing to complete and optionally download the result"""
    print(f"Waiting for processing to complete (timeout: {timeout}s)")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = check_status(api_url, session_id)
        
        if not status:
            print("Failed to get status")
            break
        
        print(f"Processing status: {status.get('status')}")
        
        if status.get("status") == "completed":
            print("\nProcessing complete!")
            print(f"People detected: {status.get('max_people_in_frame', 0)}")
            
            if download:
                output_path = download_video(api_url, session_id)
                status["downloaded_to"] = output_path
            
            return status
        
        if status.get("status") == "error":
            print(f"Processing error: {status.get('error')}")
            return status
        
        # Wait before checking again
        time.sleep(5)
    
    print("Timeout waiting for processing to complete")
    return {"session_id": session_id, "status": "timeout"}

# ----- Main Function -----
def main(args):
    # Create temp directory
    temp_dir = Path(args.output_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        setup_gpio()
        
        # ADDED: Artificial state change at the beginning
        print("*** ARTIFICIAL TRIGGER: Simulating FSR activation ***")
        force_trigger = True  # Force trigger regardless of command line args
        
        # For testing without FSR, set this to True
        if args.force_trigger:
            force_trigger = True
        
        print("Monitoring FSR. Press Ctrl+C to exit.")
        prev_state = GPIO.input(FSR_PIN)
        
        while True:
            # Check FSR state
            curr_state = GPIO.input(FSR_PIN)
            
            # Detect state change (pressure applied) or forced trigger
            if (curr_state != prev_state and curr_state == GPIO.HIGH) or force_trigger:
                print("FSR triggered! Starting image capture...")
                
                # Capture sequence of images
                image_paths = capture_image_sequence(
                    temp_dir, 
                    args.count, 
                    args.interval
                )
                
                if not image_paths:
                    print("No images captured")
                    continue
                
                # Upload images to API server
                result = upload_images(
                    args.api_url, 
                    image_paths, 
                    process_now=not args.no_process, 
                    fps=args.fps
                )
                
                if not result:
                    print("Upload failed")
                    continue
                
                session_id = result.get("session_id")
                if not session_id:
                    print("No session ID returned")
                    continue
                
                print(f"Session ID: {session_id}")
                
                # Wait for processing if requested
                if not args.no_wait and not args.no_process:
                    wait_for_processing(
                        args.api_url, 
                        session_id, 
                        timeout=args.timeout, 
                        download=not args.no_download
                    )
                
                # Reset forced trigger after one cycle
                if force_trigger:
                    force_trigger = False
            
            # Update previous state
            prev_state = curr_state
            
            # Small delay to prevent CPU hogging
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture images when FSR triggered and send to API server")
    parser.add_argument("--api-url", default="http://172.20.10.10:8000", help="API server URL")
    parser.add_argument("--output-dir", default="/tmp/person_detection", help="Directory to store captured images")
    parser.add_argument("--count", type=int, default=15, help="Number of images to capture when triggered")
    parser.add_argument("--interval", type=float, default=0.2, help="Interval between captures (seconds)")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for output video")
    parser.add_argument("--no-process", action="store_true", help="Don't process images immediately")
    parser.add_argument("--no-download", action="store_true", help="Don't download the processed video")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for processing to complete")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for processing (seconds)")
    parser.add_argument("--force-trigger", action="store_true", help="Force trigger on startup (for testing)")
    
    args = parser.parse_args()
    
    main(args)

