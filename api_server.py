from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Depends
from fastapi.responses import FileResponse
from typing import List, Optional
import uvicorn
import os
import uuid
import time
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Person Detection API")

# Directories for storing images and videos
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load YOLOv8 model
model_path = './yolov8m.pt'
model = None  # We'll load it when needed

# Store session data
sessions = {}

def get_model():
    global model
    if model is None:
        logger.info("Loading YOLOv8 model...")
        model = YOLO(model_path)
        logger.info("Model loaded successfully")
    return model

def clean_session(session_id: str):
    """Remove session data after processing"""
    if session_id in sessions:
        sessions.pop(session_id)
    
    # Remove session folder if it exists
    session_dir = UPLOAD_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)

def process_images(session_id: str, fps: int = 15):
    """Process images into a video with person detection"""
    try:
        logger.info(f"Processing images for session {session_id}")
        
        # Get session info
        if session_id not in sessions:
            logger.error(f"Session {session_id} not found")
            return None
        
        session_data = sessions[session_id]
        image_dir = UPLOAD_DIR / session_id
        
        # Check if images exist
        image_paths = sorted(list(image_dir.glob("*.jpg")))
        if not image_paths:
            logger.error(f"No images found for session {session_id}")
            return None
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Set up output file
        output_path = RESULTS_DIR / f"{session_id}_processed.mp4"
        
        # Get image dimensions from first image
        first_image = cv2.imread(str(image_paths[0]))
        if first_image is None:
            logger.error(f"Could not read image {image_paths[0]}")
            return None
        
        height, width = first_image.shape[:2]
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Get model
        model = get_model()
        
        # Process each image
        total_people = 0
        max_people_in_frame = 0
        processed_count = 0
        
        for img_path in image_paths:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Could not read image {img_path}, skipping")
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
                    filename = img_path.name
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
                
                processed_count += 1
        
        # Release resources
        video_writer.release()
        
        # Update session data
        sessions[session_id].update({
            'status': 'completed',
            'processed_count': processed_count,
            'max_people_in_frame': max_people_in_frame,
            'total_people': total_people,
            'output_path': str(output_path)
        })
        
        logger.info(f"Processing complete for session {session_id}")
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        if session_id in sessions:
            sessions[session_id]['status'] = 'error'
            sessions[session_id]['error'] = str(e)
        return None


@app.post("/api/upload/")
async def upload_images(
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    process_now: bool = Form(False),
    fps: int = Form(15)
):
    """
    Upload images for processing
    
    - session_id: Optional session ID (will be generated if not provided)
    - files: Image files to upload
    - process_now: Whether to process images immediately
    - fps: Frames per second for output video
    """
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Create session directory
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Save uploaded files
    file_count = 0
    for file in files:
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Save file with sequential numbering to ensure correct order
        file_path = session_dir / f"{file_count:05d}.jpg"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        file_count += 1
    
    # Create or update session data
    if session_id not in sessions:
        sessions[session_id] = {
            'created_at': time.time(),
            'status': 'uploaded',
            'file_count': file_count
        }
    else:
        sessions[session_id].update({
            'last_upload': time.time(),
            'status': 'uploaded',
            'file_count': sessions[session_id].get('file_count', 0) + file_count
        })
    
    # Process images if requested
    if process_now and file_count > 0:
        background_tasks.add_task(process_images, session_id, fps)
        return {
            "session_id": session_id,
            "message": f"Uploaded {file_count} files. Processing started in background.",
            "status": "processing"
        }
    
    return {
        "session_id": session_id,
        "message": f"Uploaded {file_count} files.",
        "status": "uploaded"
    }


@app.post("/api/process/{session_id}")
async def process_session(
    session_id: str, 
    background_tasks: BackgroundTasks,
    fps: int = 15
):
    """
    Process previously uploaded images
    
    - session_id: Session ID to process
    - fps: Frames per second for output video
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session_dir = UPLOAD_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail=f"No images found for session {session_id}")
    
    # Update session status
    sessions[session_id]['status'] = 'processing'
    
    # Process images in background
    background_tasks.add_task(process_images, session_id, fps)
    
    return {
        "session_id": session_id,
        "message": "Processing started in background",
        "status": "processing"
    }


@app.get("/api/status/{session_id}")
async def check_status(session_id: str):
    """Check the status of a processing session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return {
        "session_id": session_id,
        "status": sessions[session_id]['status'],
        **{k: v for k, v in sessions[session_id].items() if k not in ['status']}
    }


@app.get("/api/download/{session_id}")
async def download_video(session_id: str):
    """Download the processed video"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session_data = sessions[session_id]
    if session_data.get('status') != 'completed':
        raise HTTPException(status_code=400, detail=f"Processing for session {session_id} is not complete")
    
    output_path = session_data.get('output_path')
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail=f"Output video for session {session_id} not found")
    
    return FileResponse(
        path=output_path,
        filename=f"processed_{session_id}.mp4",
        media_type="video/mp4"
    )


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str, background_tasks: BackgroundTasks):
    """Delete a session and its data"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Get output path before cleaning session
    output_path = sessions[session_id].get('output_path')
    
    # Clean up session data in background
    background_tasks.add_task(clean_session, session_id)
    
    # Delete output file if it exists
    if output_path and os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception as e:
            logger.error(f"Error deleting output file: {str(e)}")
    
    return {"message": f"Session {session_id} deleted"}


@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": [
            {
                "session_id": session_id,
                "status": data['status'],
                "created_at": data.get('created_at'),
                "file_count": data.get('file_count', 0)
            }
            for session_id, data in sessions.items()
        ]
    }


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 