# Video Caption Generator

This Streamlit application allows you to upload videos or download YouTube videos and generate captions for individual frames using multiple AI models (GPT-4, Claude, and Gemini). The app extracts frames from the video at specified intervals and lets you navigate through them to generate and compare captions from different AI providers.

## Features

- Video upload and YouTube video download
- Frame extraction from videos
- Frame-by-frame navigation
- Caption generation using multiple AI models:
  - GPT-4 Vision
  - Claude 3 Opus
  - Google Gemini Pro Vision
- Side-by-side comparison of captions
- Adjustable frame extraction interval

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure API keys using one of these methods:

   a. Using a `.env` file (recommended for development):
   Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

   b. Using environment variables (recommended for production):
   Set the environment variables in your terminal:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   export ANTHROPIC_API_KEY=your_anthropic_api_key
   export GOOGLE_API_KEY=your_google_api_key
   ```

   Note: Environment variables take precedence over values in the `.env` file.

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Choose your input method:
   - Upload a video file (supported formats: MP4, AVI, MOV)
   - Enter a YouTube URL and click "Download Video"
3. Adjust the frame extraction interval if needed
4. Click "Extract Frames" to process the video
5. Navigate through the frames using the "Previous Frame" and "Next Frame" buttons
6. Click "Generate Captions" to get captions from all AI models for the current frame
7. Use the "Clear Video" button to remove the current video and start over

## Requirements

- Python 3.8+
- Streamlit
- OpenCV
- PyTube (for YouTube downloads)
- OpenAI API key
- Anthropic API key
- Google API key

## Notes

- The app uses temporary files and directories for processing, which are automatically cleaned up
- API calls may incur costs depending on your usage and the providers' pricing
- Frame extraction interval can be adjusted to balance between detail and processing time
- Environment variables can be set either through a `.env` file or directly in the terminal
- YouTube downloads are saved temporarily and will be deleted when you clear the video or close the app 