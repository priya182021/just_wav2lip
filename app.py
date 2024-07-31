
import gradio as gr
import subprocess
from subprocess import call

with gr.Blocks() as ui:
    with gr.Row():
        video = gr.File(label="Video or Image")
        audio = gr.File(label="Audio")
        with gr.Column():
            checkpoint = gr.Radio(["wav2lip", "wav2lip_gan"], label="Checkpoint")
            no_smooth = gr.Checkbox(label="No Smooth")
            resize_factor = gr.Slider(minimum=1, maximum=4, step=1, label="Resize Factor")
    with gr.Row():
        with gr.Column():
            pad_top = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Top")
            pad_bottom = gr.Slider(minimum=0, maximum=50, step=1, value=10, label="Pad Bottom (Often increasing this to 20 allows chin to be included)")
            pad_left = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Left")
            pad_right = gr.Slider(minimum=0, maximum=50, step=1, value=0, label="Pad Right")
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
    
ui.queue().launch(share=True,debug=True)
