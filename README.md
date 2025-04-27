# Person Detection API System

This system allows you to capture images from a Raspberry Pi camera, send them to a server for processing, and receive an annotated video showing detected people.

## System Components

1. **API Server** - Runs on a computer/server and processes images for person detection
2. **Raspberry Pi Client** - Captures images and sends them to the API server

## Setup Instructions

### Server Setup (Computer)

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLOv8 model files (yolov8m.pt should already be included in this repo):
   ```bash
   # If model is not included, run:
   # wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
   ```

4. Start the FastAPI server:
   ```bash
   python api_server.py
   ```

   The server will start on http://0.0.0.0:8000 by default.

5. You can access the API documentation at http://localhost:8000/docs

### Raspberry Pi Setup

1. Clone this repository on your Raspberry Pi:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
   # Also install picamera2 (Raspberry Pi specific)
   pip install picamera2
   ```

3. Test the client:
   ```bash
   python raspberry_pi_client.py --api-url http://<server-ip>:8000
   ```

   This will capture 5 images by default and send them to the server for processing.

## Usage

### Raspberry Pi Client

Basic usage:
```bash
python raspberry_pi_client.py --api-url http://<server-ip>:8000
```

Advanced options:
```bash
python raspberry_pi_client.py \
  --api-url http://<server-ip>:8000 \
  --output-dir ./captured_images \
  --count 10 \                    # Number of images to capture
  --interval 0.5 \                # Interval between captures (seconds)
  --width 1280 \                  # Image width
  --height 720 \                  # Image height
  --fps 15                        # Frames per second for output video
```

Additional flags:
- `--no-process`: Don't process images immediately
- `--no-download`: Don't download the processed video
- `--no-wait`: Don't wait for processing to complete
- `--timeout 300`: Timeout for processing (seconds)

### API Endpoints

- `POST /api/upload/`: Upload images for processing
- `POST /api/process/{session_id}`: Process previously uploaded images
- `GET /api/status/{session_id}`: Check the status of a processing session
- `GET /api/download/{session_id}`: Download the processed video
- `DELETE /api/session/{session_id}`: Delete a session and its data
- `GET /api/sessions`: List all active sessions

## How It Works

1. **Image Capture**: The Raspberry Pi captures a sequence of images using its camera.
2. **Upload**: Images are uploaded to the server via the API.
3. **Processing**: The server uses YOLOv8 to detect people in the images and creates an annotated video.
4. **Result**: The processed video is made available for download.

## System Flow

```
┌────────────────┐     ┌───────────────────┐     ┌────────────────┐
│  Raspberry Pi  │────▶│  API Server       │────▶│ Processed      │
│  (captures     │     │  (detects people, │     │ Video          │
│   images)      │     │   creates video)  │     │                │
└────────────────┘     └───────────────────┘     └────────────────┘
```

## Troubleshooting

- **Camera not working**: Make sure the Raspberry Pi camera is enabled and properly connected.
- **Connection errors**: Verify that the server IP address is correct and that the server is running.
- **Processing errors**: Check the server logs for details.

## License

MIT