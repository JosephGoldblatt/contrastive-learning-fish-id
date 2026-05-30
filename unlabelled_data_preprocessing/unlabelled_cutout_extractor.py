#!/usr/bin/env python3

"""
Processes all videos in a directory. Samples frames from videos at one second intervals.
Excludes the first ten and last five minutes of each video (to lower chance of accidentally detecting object above surface)
Extracts "cutouts" of fish identified from frames using a YOLO detector (with confidence<85%).
Stores these cutouts in a separate directory.

Usage: python3 unlabelled_cutout_extractor.py <root_directory> <interval> <model_path> [confidence]
"""

import sys
import subprocess
import shutil
from pathlib import Path
from PIL import Image
from ultralytics import YOLO #type: ignore


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', 
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return None


def extract_frames(video_dir, interval):
    """Extract frames from all videos in directory, excluding first 10 minutes and last 5 minutes."""
    mp4_files = []
    for file_path in Path(video_dir).rglob("*L.avi"):
        if file_path.is_file():
            mp4_files.append(file_path)
    for file_path in Path(video_dir).rglob("*L.AVI"):
        if file_path.is_file():
            mp4_files.append(file_path)
    
    if not mp4_files:
        return None
    
    # Create output directory
    output_dir =f"extracted_unlabelled_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Time exclusions
    start_skip = 600  # 10 minutes
    end_skip = 300    # 5 minutes
    
    for video_path in mp4_files:
        video_name = video_path.stem
        video_frames_dir = output_dir / f"{video_name}_frames"
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        
        duration = get_video_duration(video_path)
        if duration is None:
            continue
        
        effective_duration = duration - start_skip - end_skip
        if effective_duration <= 0:
            continue
        
        fps_value = 1.0 / interval
        output_pattern = video_frames_dir / f"{video_name}_frame_%d.jpg"
        
        cmd = [
            'ffmpeg', '-ss', str(start_skip), '-i', str(video_path), 
            '-t', str(effective_duration), '-vf', f'fps={fps_value}', 
            '-y', str(output_pattern)
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return output_dir


def run_yolo_detection(frames_dir, model_path, confidence):
    """Run YOLO detection on all frames and save predictions."""
    model = YOLO(model_path)
    
    for subdir in frames_dir.iterdir():
        if subdir.is_dir() and subdir.name.endswith('_frames'):
            for frame_path in subdir.glob("*_frame_*.jpg"):
                results = model(str(frame_path), conf=confidence, verbose=False)
                
                txt_path = frame_path.with_suffix('.txt')
                with open(txt_path, 'w') as f:
                    if results[0].boxes is not None:
                        img_height, img_width = results[0].orig_shape
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = ((x1 + x2) / 2) / img_width
                            center_y = ((y1 + y2) / 2) / img_height
                            width = (x2 - x1) / img_width
                            height = (y2 - y1) / img_height
                            class_id = int(box.cls[0].cpu().numpy())
                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")


def extract_cutouts(frames_dir):
    """Extract cutouts from frames based on YOLO predictions."""
    output_dir = frames_dir / "extracted_cutouts"
    output_dir.mkdir(exist_ok=True)
    
    cutout_count = 0
    
    for subdir in frames_dir.iterdir():
        if subdir.is_dir() and subdir.name.endswith('_frames'):
            for jpg_file in subdir.glob("*_frame_*.jpg"):
                txt_file = jpg_file.with_suffix('.txt')
                if not txt_file.exists():
                    continue
                
                image = Image.open(jpg_file)
                img_width, img_height = image.size
                base_name = jpg_file.stem
                
                with open(txt_file, 'r') as f:
                    for i, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            center_x, center_y, width, height = map(float, parts[1:5])
                            
                            # Convert to pixel coordinates
                            x1 = int((center_x - width/2) * img_width)
                            y1 = int((center_y - height/2) * img_height)
                            x2 = int((center_x + width/2) * img_width)
                            y2 = int((center_y + height/2) * img_height)
                            
                            # Clamp to image bounds
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(img_width, x2), min(img_height, y2)
                            
                            if x2 > x1 and y2 > y1:
                                cutout = image.crop((x1, y1, x2, y2))
                                cutout_path = output_dir / f"{base_name}_cutout_{i}.jpg"
                                cutout.save(cutout_path)
                                cutout_count += 1
    
    return output_dir, cutout_count


def main():
    if len(sys.argv) < 4:
        print("Usage: python process_videos.py <root_directory> <interval> <model_path> [confidence]")
        sys.exit(1)
    
    root_dir = Path(sys.argv[1])
    interval = float(sys.argv[2])
    model_path = sys.argv[3]
    confidence = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    
    total_cutouts = 0
    
    for subdir in root_dir.iterdir():
        if subdir.is_dir():
            frames_dir = extract_frames(subdir, interval)
            if frames_dir is None:
                continue
            
            run_yolo_detection(frames_dir, model_path, confidence)
            cutouts_dir, cutout_count = extract_cutouts(frames_dir)
            
            if cutout_count == 0:
                shutil.rmtree(frames_dir)
                continue
            
            total_cutouts += cutout_count
            
            # Clean up frames, keep cutouts
            for subdir_frame in frames_dir.iterdir():
                if subdir_frame.is_dir() and subdir_frame.name.endswith('_frames'):
                    shutil.rmtree(subdir_frame)
    
    print(f"Extracted {total_cutouts} fish cutouts")


if __name__ == "__main__":
    main()