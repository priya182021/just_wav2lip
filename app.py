Hugging Face's logo
Hugging Face
Search models, datasets, users...
Models
Datasets
Spaces
Posts
Docs
Pricing



Spaces:

priyankad199
/
Wav2lip





like
44
App
Files
Community
2
gradio-lipsync-wav2lip
/
app.py



Copy download link
history
blame
contribute
delete
No virus

2.39 kB
import gradio as gr
import subprocess
from subprocess import call

with gr.Blocks() as ui:
    with gr.Row():
        video = gr.File(label="Video or Image", info="Filepath of video/image that contains faces to use")
        audio = gr.File(label="Audio", info="Filepath of video/audio file to use as raw audio source")
        with gr.Column():
            checkpoint = gr.Radio(["wav2lip", "wav2lip_gan"], label="Checkpoint", info="Name of saved checkpoint to load weights from")
            no_smooth = gr.Checkbox(label="No Smooth", info="Prevent smoothing face detections over a short temporal window")
            resize_factor = gr.Slider(minimum=1, maximum=4, step=1, label="Resize Factor", info="Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p")
    with gr.Row():
        with gr.Column():
            pad_top = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Top", info="Padding above")
            pad_bottom = gr.Slider(minimum=0, maximum=50, step=1, value=10, label="Pad Bottom (Often increasing this to 20 allows chin to be included)", info="Padding below lips")
            pad_left = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Left", info="Padding to the left of lips")
            pad_right = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Right", info="Padding to the right of lips")
            generate_btn = gr.Button("Generate")
        with gr.Column():
           result = gr.Video()

    def generate(video, audio, checkpoint, no_smooth, resize_factor, pad_top, pad_bottom, pad_left, pad_right):
        if video is None or audio is None or checkpoint is None:
            return
        
        smooth = "--nosmooth" if no_smooth else ""
        
        
        cmd = [
            "python",
            "inference.py",
            "--checkpoint_path", f"checkpoints/{checkpoint}.pth",
            "--segmentation_path", "checkpoints/face_segmentation.pth",
            "--enhance_face", "gfpgan",
            "--face", video.name,
            "--audio", audio.name,
            "--outfile", "results/output.mp4",
        ]

        call(cmd)
        return "results/output.mp4"

    generate_btn.click(
        generate, 
        [video, audio, checkpoint, pad_top, pad_bottom, pad_left, pad_right, resize_factor], 
        result)
    
ui.queue().launch(debug=True)
