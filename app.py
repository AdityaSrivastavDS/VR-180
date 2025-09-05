
import streamlit as st
from depth_processing import convert_video_to_vr180, sample_preview_frames
import tempfile, os
from pathlib import Path

st.set_page_config(page_title='VR-180 Converter', layout='wide')

st.title('🎬 2D → VR-180 Converter (Streamlit MVP)')
st.markdown('Upload a 2D MP4 clip and convert it into a stereoscopic VR-180 side-by-side MP4. This MVP uses MiDaS depth estimation and a simple view-synthesis method.')

with st.expander('⚙️ Settings (advanced)'):
    max_width = st.number_input('Processing width (px, downscale for speed)', value=960, min_value=320, max_value=1920, step=32)
    frame_step = st.number_input('Frame step (process every Nth frame)', value=1, min_value=1, max_value=10)
    shift_pixels = st.number_input('Max horizontal parallax (pixels)', value=40, min_value=1, max_value=300)
    use_fast_midas = st.checkbox('Use faster (smaller) MiDaS model if available (faster but lower quality)', value=True)

uploaded = st.file_uploader('Upload 2D MP4 video', type=['mp4', 'mov', 'avi', 'mkv'])

if uploaded is not None:
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, uploaded.name)
    with open(input_path, 'wb') as f:
        f.write(uploaded.getbuffer())

    st.video(input_path)

    st.markdown('---')
    st.write('Preview (small sample frames from the clip):')
    preview_cols = st.columns(2)
    preview_paths = sample_preview_frames(input_path, max_width=max_width)
    if preview_paths:
        with preview_cols[0]:
            st.image(preview_paths[0], caption='Frame (original)')
        with preview_cols[1]:
            st.image(preview_paths[1], caption='Depth map (normalized)')

    if st.button('Convert to VR-180 (start)'):
        status_text = st.empty()
        progress_bar = st.progress(0)
        out_file = os.path.join(tmp_dir, Path(input_path).stem + '_vr180_side_by_side.mp4')

        try:
            convert_video_to_vr180(
                input_path,
                out_file,
                max_width=int(max_width),
                frame_step=int(frame_step),
                max_shift=int(shift_pixels),
                use_fast_midas=bool(use_fast_midas),
                progress_callback=lambda p, msg=None: (progress_bar.progress(int(p*100)), status_text.text(msg or f'Progress: {int(p*100)}%'))
            )
            st.success('Conversion finished! Download below.')
            st.video(out_file)
            with open(out_file, 'rb') as f:
                st.download_button('Download VR-180 side-by-side MP4', data=f, file_name=os.path.basename(out_file), mime='video/mp4')
        except Exception as e:
            st.error(f'Processing failed: {e}')
