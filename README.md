
# VR-180 Converter (Streamlit MVP)

This is an MVP Streamlit app that converts a 2D video clip into a stereoscopic VR-180 side-by-side video using **monocular depth estimation** (MiDaS) + simple view synthesis.

**Important notes before running**
- This project uses PyTorch and the MiDaS depth estimation model which will be downloaded automatically the first time you run the app (internet required).
- GPU is recommended for speed but CPU will work (slower).
- For large videos, reduce resolution and/or frame rate to speed up processing (there are settings in the UI).

## What is included
- `app.py` - Main Streamlit interface.
- `depth_processing.py` - Video frame extraction, depth estimation, stereo synthesis, and video writing.
- `requirements.txt` - Python dependencies.
- `scripts/` - helper script examples (if needed).
- `LICENSE` - MIT

## How to run (local)
1. Create a new Python environment (conda or venv recommended)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Upload a 2D clip (MP4) and click **Convert**.
5. Download the resulting side-by-side VR-180 MP4 file when processing completes.

## Tips for best results
- Use a short clip for testing (5-15 seconds).
- Lower the resolution (e.g., width=960) to speed up processing.
- If you have a GPU, ensure PyTorch installs with CUDA matching your drivers.
- The conversion uses a simple horizontal parallax shift derived from depth maps; it's an MVP approach — results vary with scene complexity.
