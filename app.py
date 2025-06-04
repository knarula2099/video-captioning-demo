import streamlit as st
import os
import tempfile
from video_utils import extract_frames, get_video_info
from caption_generator import CaptionGenerator
import shutil
from PIL import Image as PIL_Image
import json
import pandas as pd
from datetime import datetime
import base64

# Set page config
st.set_page_config(
    page_title="Video Caption Generator",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'processed_frames' not in st.session_state:
    st.session_state.processed_frames = []
if 'captions' not in st.session_state:
    st.session_state.captions = []
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = ["Gemini"]
if 'custom_prompt' not in st.session_state:
    st.session_state.custom_prompt = None

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def cleanup_files():
    """Clean up all generated files."""
    # Remove video file if it exists
    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        os.unlink(st.session_state.video_path)

    # Remove frames directory if it exists
    if os.path.exists("frames"):
        shutil.rmtree("frames")

    # Reset session state
    st.session_state.video_path = None
    st.session_state.processed_frames = []
    st.session_state.captions = []

def create_caption_dataframe():
    """Create a pandas DataFrame from captions for better visualization."""
    if not st.session_state.processed_frames or not st.session_state.captions:
        return None
        
    data = []
    for idx, (fp, caption_dict) in enumerate(zip(st.session_state.processed_frames, st.session_state.captions), 1):
        row = {
            "Frame_Number": idx,
            "Frame_Path": os.path.basename(fp),
            "Full_Frame_Path": fp,
        }
        
        # Add caption columns for each model
        for model, caption in caption_dict.items():
            row[f"{model}_Caption"] = caption
            
        data.append(row)
    
    return pd.DataFrame(data)

def export_captions_csv():
    """Export captions to CSV format."""
    df = create_caption_dataframe()
    if df is None:
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_content = df.to_csv(index=False)
    return csv_content, f"video_captions_{timestamp}.csv"

def export_captions_json():
    """Export captions to JSON format."""
    if not st.session_state.processed_frames or not st.session_state.captions:
        return None
        
    # Create export data
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "video_path": st.session_state.video_path,
        "frames": []
    }
    
    for fp, caption_dict in zip(st.session_state.processed_frames, st.session_state.captions):
        frame_data = {
            "frame_path": fp,
            "captions": caption_dict
        }
        export_data["frames"].append(frame_data)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_content = json.dumps(export_data, indent=2)
    return json_content, f"video_captions_{timestamp}.json"

def export_captions_html():
    """Export captions with images to HTML format."""
    if not st.session_state.processed_frames or not st.session_state.captions:
        return None
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Video Caption Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 20px;
            }}
            .frame-item {{
                margin-bottom: 40px;
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                background-color: #fafafa;
            }}
            .frame-header {{
                background-color: #4CAF50;
                color: white;
                padding: 15px;
                font-weight: bold;
                font-size: 18px;
            }}
            .frame-content {{
                display: flex;
                padding: 20px;
                gap: 20px;
            }}
            .frame-image {{
                flex: 0 0 300px;
            }}
            .frame-image img {{
                width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            .frame-captions {{
                flex: 1;
            }}
            .caption-item {{
                margin-bottom: 15px;
                padding: 15px;
                background-color: white;
                border-radius: 5px;
                border-left: 4px solid #4CAF50;
            }}
            .model-name {{
                font-weight: bold;
                color: #2196F3;
                margin-bottom: 8px;
                font-size: 16px;
            }}
            .caption-text {{
                color: #333;
                line-height: 1.5;
                font-size: 14px;
            }}
            .summary {{
                background-color: #e8f5e8;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
            }}
            @media (max-width: 768px) {{
                .frame-content {{
                    flex-direction: column;
                }}
                .frame-image {{
                    flex: none;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎥 Video Caption Report</h1>
                <p>Generated on {timestamp}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Prompt Used:</strong> {st.session_state.custom_prompt or "Describe this image in one concise sentence (present simple). Use two sentences only if truly needed (<10% of cases)."}</p>
                <p><strong>Total Frames:</strong> {len(st.session_state.processed_frames)}</p>
                <p><strong>Models Used:</strong> {', '.join(set().union(*[caption_dict.keys() for caption_dict in st.session_state.captions]))}</p>
            </div>
    """
    
    # Add each frame with its captions
    for idx, (fp, caption_dict) in enumerate(zip(st.session_state.processed_frames, st.session_state.captions), 1):
        # Convert image to base64 for embedding
        try:
            with open(fp, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                img_tag = f'<img src="data:image/jpeg;base64,{img_base64}" alt="Frame {idx}">'
        except Exception:
            img_tag = f'<p>Image not found: {os.path.basename(fp)}</p>'
        
        html_content += f"""
            <div class="frame-item">
                <div class="frame-header">
                    Frame {idx} - {os.path.basename(fp)}
                </div>
                <div class="frame-content">
                    <div class="frame-image">
                        {img_tag}
                    </div>
                    <div class="frame-captions">
        """
        
        # Add captions for each model
        for model, caption in caption_dict.items():
            html_content += f"""
                        <div class="caption-item">
                            <div class="model-name">{model}</div>
                            <div class="caption-text">{caption}</div>
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    return html_content, f"video_captions_report_{timestamp_file}.html"

def main():
    st.title("🎥 Video Caption Generator")
    st.write("Upload a video or provide a YouTube URL to automatically generate captions for all extracted frames")

    # Model selection
    model_options = ["All", "Gemini", "Claude", "GPT-4"]
    
    selected_model_option = st.selectbox(
        "Choose AI Model",
        model_options,
        index=0,  # Default to "All"
        help="Select which AI model(s) to use for caption generation"
    )
    
    # Convert selection to list of models for compatibility
    if selected_model_option == "All":
        st.session_state.selected_models = ["Gemini", "Claude", "GPT-4"]
    else:
        st.session_state.selected_models = [selected_model_option]
    
    # Show selected models
    if selected_model_option == "All":
        st.info("Using all available models: Gemini, Claude, and GPT-4")
    else:
        st.info(f"Using: {selected_model_option}")
        
    # Check for missing API keys and warn user
    missing_keys = []
    if not GOOGLE_API_KEY and "Gemini" in st.session_state.selected_models:
        missing_keys.append("GOOGLE_API_KEY")
    if not ANTHROPIC_API_KEY and "Claude" in st.session_state.selected_models:
        missing_keys.append("ANTHROPIC_API_KEY") 
    if not OPENAI_API_KEY and "GPT-4" in st.session_state.selected_models:
        missing_keys.append("OPENAI_API_KEY")
        
    if missing_keys:
        st.warning(f"Missing API keys: {', '.join(missing_keys)}. These models will be skipped.")

    # Custom prompt
    st.session_state.custom_prompt = st.text_area(
        "Custom Prompt (optional)",
        value=st.session_state.custom_prompt,
        help="Leave empty to use the default prompt"
    )

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.video_path = tmp_file.name

    # Process video if available
    if st.session_state.video_path:
        # Show video info
        width, height, fps = get_video_info(st.session_state.video_path)
        st.write(f"Video Info: {width}x{height} @ {fps:.2f} fps")

        # Frame extraction settings
        col1, col2 = st.columns(2)
        with col1:
            frame_interval = st.slider("Extract frame every N frames", 1, 60, 30)
        with col2:
            if st.button("Extract Frames and Generate Captions"):
                with st.spinner("Extracting frames..."):
                    # Extract frames from local file
                    frame_paths = extract_frames(st.session_state.video_path, "frames", frame_interval)
                    st.session_state.processed_frames = frame_paths

                if frame_paths:
                    st.write("Generating captions for all frames...")
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    caption_generator = CaptionGenerator(
                        google_api_key=GOOGLE_API_KEY,
                        anthropic_api_key=ANTHROPIC_API_KEY,
                        openai_api_key=OPENAI_API_KEY,
                        default_prompt=st.session_state.custom_prompt
                    )
                    
                    captions_list = []
                    total_frames = len(frame_paths)
                    
                    for idx, fp in enumerate(frame_paths):
                        # Update progress bar and status
                        progress = (idx + 1) / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {idx + 1} of {total_frames}")
                        
                        captions = caption_generator.generate_all_captions(fp)
                        # Filter captions based on selected models
                        filtered_captions = {
                            model: caption 
                            for model, caption in captions.items() 
                            if model in st.session_state.selected_models
                        }
                        captions_list.append(filtered_captions)
                    
                    # Complete the progress bar
                    progress_bar.progress(1.0)
                    status_text.text(f"✅ Completed! Generated captions for {total_frames} frames")
                    
                    st.session_state.captions = captions_list

                    st.success(f"Extracted {len(frame_paths)} frames and generated captions!")

        # Display all frames with captions
        if st.session_state.processed_frames and st.session_state.captions:
            st.markdown("### Frames with Captions")
            
            # Export options
            st.markdown("#### Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV Download
                csv_content, csv_filename = export_captions_csv()
                if csv_content:
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_content,
                        file_name=csv_filename,
                        mime="text/csv",
                        help="Download as CSV for Excel/Google Sheets"
                    )
            
            with col2:
                # JSON Download
                json_content, json_filename = export_captions_json()
                if json_content:
                    st.download_button(
                        label="📋 Download JSON",
                        data=json_content,
                        file_name=json_filename,
                        mime="application/json",
                        help="Download as JSON for developers"
                    )
            
            with col3:
                # HTML Report Download
                html_content, html_filename = export_captions_html()
                if html_content:
                    st.download_button(
                        label="🌐 Download HTML Report",
                        data=html_content,
                        file_name=html_filename,
                        mime="text/html",
                        help="Download visual report with images and captions"
                    )
            

            
            for fp, caption_dict in zip(st.session_state.processed_frames, st.session_state.captions):
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image(fp, width=300)
                with cols[1]:
                    for model, caption in caption_dict.items():
                        st.write(f"**{model}:** {caption}")
                st.markdown("---")

        # Clean up button
        if st.button("Clear Video and Frames"):
            cleanup_files()
            st.rerun()

if __name__ == "__main__":
    main()
