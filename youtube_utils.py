import yt_dlp
import os
import cv2
import numpy as np
from typing import Tuple, List

def download_youtube_video(url: str) -> Tuple[str, str]:
    """
    Download a YouTube video and return the path to the downloaded file.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        Tuple[str, str]: (video_path, video_title)
    """
    try:
        # Create downloads directory if it doesn't exist
        downloads_dir = os.path.join(os.getcwd(), "downloads")
        if not os.path.exists(downloads_dir):
            os.makedirs(downloads_dir)
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]',  # Best quality MP4
            'outtmpl': os.path.join(downloads_dir, '%(title)s.%(ext)s'),  # Output template
            'quiet': True,  # Suppress output
            'no_warnings': True,  # Suppress warnings
        }
        
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Unknown Title')
            
            # Download the video
            ydl.download([url])
            
            # Get the actual file path
            video_path = os.path.join(downloads_dir, f"{video_title}.mp4")
        
        return video_path, video_title
        
    except Exception as e:
        raise Exception(f"Error downloading YouTube video: {str(e)}")

def get_youtube_stream_info(url: str) -> Tuple[str, str, int, int, float]:
    """
    Get YouTube video stream URL and metadata without downloading.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        Tuple[str, str, int, int, float]: (stream_url, video_title, width, height, fps)
    """
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            video_title = info.get('title', 'Unknown Title')
            stream_url = info.get('url')
            width = info.get('width', 0)
            height = info.get('height', 0)
            fps = info.get('fps', 0.0)
            
            if not stream_url:
                raise Exception("Could not extract stream URL")
                
            return stream_url, video_title, width, height, fps
            
    except Exception as e:
        raise Exception(f"Error getting YouTube stream info: {str(e)}")

def extract_frames_from_youtube_stream(url: str, output_dir: str = "frames", frame_interval: int = 30) -> Tuple[List[str], str]:
    """
    Extract frames directly from a YouTube video stream without downloading the full video.
    
    Args:
        url (str): YouTube video URL
        output_dir (str): Directory to save extracted frames
        frame_interval (int): Check every Nth frame for similarity
        
    Returns:
        Tuple[List[str], str]: (List of frame paths, video title)
    """
    # Get stream URL and info
    stream_url, video_title, width, height, fps = get_youtube_stream_info(url)
    
    # Same histogram-based frame extraction logic as the original
    HIST_THRESHOLD = 0.8
    
    # Create (or clear) the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for f in os.listdir(output_dir):
            if f.startswith("frame_") and f.endswith(".jpg"):
                os.remove(os.path.join(output_dir, f))
    
    # Try to open video stream directly first
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        # Fallback: Try with additional OpenCV backend options
        print("Direct stream failed, trying with CAP_FFMPEG backend...")
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        
    if not cap.isOpened():
        # Final fallback: Download a temporary small portion for frame extraction
        print("Stream access failed, falling back to partial download...")
        return extract_frames_from_youtube_fallback(url, output_dir, frame_interval)
    
    frame_paths: List[str] = []
    last_saved_hist: np.ndarray = None
    frame_count = 0
    saved_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every `frame_interval` frames
            if frame_count % frame_interval == 0:
                # Compute a normalized grayscale histogram (256 bins)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                # If this is the first sampled frame, save it unconditionally
                if last_saved_hist is None:
                    save_flag = True
                else:
                    # Compute χ² distance between last_saved_hist and current hist
                    chi_sq = cv2.compareHist(last_saved_hist.astype(np.float32),
                                           hist.astype(np.float32),
                                           method=cv2.HISTCMP_CHISQR)
                    save_flag = (chi_sq > HIST_THRESHOLD)
                
                if save_flag:
                    filename = f"frame_{frame_count:06d}.jpg"
                    out_path = os.path.join(output_dir, filename)
                    cv2.imwrite(out_path, frame)
                    frame_paths.append(out_path)
                    
                    last_saved_hist = hist.copy()
                    saved_count += 1
            
            frame_count += 1
    
    finally:
        cap.release()
    
    print(f"Scanned {frame_count} frames, saved {saved_count} 'interesting' frames from stream.")
    return frame_paths, video_title

def extract_frames_from_youtube_fallback(url: str, output_dir: str = "frames", frame_interval: int = 30) -> Tuple[List[str], str]:
    """
    Fallback method: Download a small portion of the video to extract frames when streaming fails.
    """
    try:
        import tempfile
        
        # Download just the first few MB of the video for frame extraction
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'quiet': True,
            'no_warnings': True,
            'external_downloader': 'ffmpeg',
            'external_downloader_args': ['-ss', '0', '-t', '60'],  # First 60 seconds only
        }
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            ydl_opts['outtmpl'] = temp_path
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'Unknown Title')
                ydl.download([url])
            
            # Now extract frames from the temporary file
            from video_utils import extract_frames
            frame_paths = extract_frames(temp_path, output_dir, frame_interval)
            
            return frame_paths, video_title
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        raise Exception(f"Both streaming and fallback download failed: {str(e)}")

def get_youtube_video_info(url: str) -> Tuple[int, int, float]:
    """
    Get basic video information from YouTube URL without downloading.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        Tuple[int, int, float]: (width, height, fps)
    """
    try:
        _, _, width, height, fps = get_youtube_stream_info(url)
        return width, height, fps
    except Exception as e:
        raise Exception(f"Error getting video info: {str(e)}")

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=csqJNlg_v2A&t=19s"
    
    # Test streaming extraction
    print("Testing stream extraction...")
    frame_paths, title = extract_frames_from_youtube_stream(url)
    print(f"Extracted {len(frame_paths)} frames from: {title}")
    
    # Test video info
    print("Testing video info...")
    width, height, fps = get_youtube_video_info(url)
    print(f"Video info: {width}x{height} @ {fps:.2f} fps")