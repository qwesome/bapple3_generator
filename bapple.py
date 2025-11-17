import cv2
import numpy as np
from tqdm import tqdm
import subprocess
import os

def extract_audio(input_path, audio_path):
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vn',
            '-acodec', 'copy',
            '-y',
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def merge_audio_video(video_path, audio_path, output_path):
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-y',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def process_video(input_path, output_path, grid_size=24):
    temp_audio = "temp_audio.aac"
    temp_video = "temp_video.mp4"
    
    has_audio = extract_audio(input_path, temp_audio)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    cell_width = width // grid_size
    cell_height = height // grid_size
    
    output_file = temp_video if has_audio else output_path
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        return
    
    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break
        
        downscaled = cv2.resize(frame, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        inverted_downscaled = 255 - downscaled
        
        output_frame = np.zeros_like(frame)
        
        for row in range(grid_size):
            for col in range(grid_size):
                y_start = row * cell_height
                y_end = (row + 1) * cell_height
                x_start = col * cell_width
                x_end = (col + 1) * cell_width
                
                target_region = frame[y_start:y_end, x_start:x_end]
                
                if target_region.shape[:2] != (cell_height, cell_width):
                    target_region = cv2.resize(target_region, (cell_width, cell_height))
                
                error_normal = np.sum((target_region.astype(float) - downscaled.astype(float)) ** 2)
                error_inverted = np.sum((target_region.astype(float) - inverted_downscaled.astype(float)) ** 2)
                
                if error_normal < error_inverted:
                    output_frame[y_start:y_end, x_start:x_end] = downscaled
                else:
                    output_frame[y_start:y_end, x_start:x_end] = inverted_downscaled
        
        out.write(output_frame)
    
    cap.release()
    out.release()
    
    if has_audio:
        if merge_audio_video(temp_video, temp_audio, output_path):
            try:
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
            except:
                pass

if __name__ == "__main__":
    try:
        process_video("video.mp4", "output.mp4", grid_size=24)
    except:
        pass