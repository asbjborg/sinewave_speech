"""
Gradio Web UI for Sinewave Speech Synthesis
"""

import gradio as gr
import numpy as np
import tempfile
import os
from pathlib import Path
from sws import (
    load_wave,
    bp_filter_and_decimate,
    normalize,
    sinethesise,
    lpc_vocode,
    modfm_buzz,
    upsample
)


def process_audio(
    audio_file,
    order,
    low_freq,
    high_freq,
    decimation,
    window_size,
    overlap,
    bw_amp,
    mode,
    buzz_freq,
    create_comparison,
    gap_duration
):
    """
    Process audio file with sinewave speech synthesis
    
    Returns: (sample_rate, processed_audio_array, comparison_audio_array or None)
    """
    try:
        if audio_file is None:
            return None, None, "Please upload an audio file"
        
        # Load the audio file
        wav, fs = load_wave(audio_file)
        
        # Filter and decimate
        wav_filtered = normalize(bp_filter_and_decimate(
            wav, low_freq, high_freq, fs, decimate=decimation
        ))
        
        lpc_order = 2 * order + 2
        
        # Process based on mode
        if mode == "Sinewave":
            modulated = sinethesise(
                wav_filtered,
                frame_len=window_size,
                order=lpc_order,
                use_lsp=True,
                bw_amp=bw_amp,
                sr=fs / decimation,
                noise=0.0,
                overlap=overlap
            )
        elif mode == "Buzz":
            N = 12 * np.log2(float(buzz_freq)/440.0) + 69            
            k = np.exp(-0.1513*N) + 15.927
            carrier = modfm_buzz(
                len(wav_filtered), 
                f=np.full(len(wav_filtered), buzz_freq, dtype=np.float64),
                sr=float(fs/decimation), 
                k=np.full(len(wav_filtered), k*k)
            )
            modulated = lpc_vocode(
                wav_filtered, 
                frame_len=window_size, 
                order=lpc_order,
                carrier=carrier, 
                residual_amp=0.0, 
                vocode_amp=1, 
                env=True, 
                freq_shift=1
            )
        elif mode == "Noise":
            carrier = np.random.normal(0, 1, len(wav_filtered))
            modulated = lpc_vocode(
                wav_filtered, 
                frame_len=window_size, 
                order=lpc_order,
                carrier=carrier, 
                residual_amp=0, 
                vocode_amp=1, 
                env=True, 
                freq_shift=1
            )
        
        # Upsample and normalize
        up_modulated = normalize(upsample(modulated, decimation))
        
        # Prepare output audio
        output_audio = (fs, (up_modulated * 32767.0).astype(np.int16))
        
        # Create comparison if requested
        comparison_audio = None
        if create_comparison:
            gap_samples = int(gap_duration * fs)
            gap = np.zeros(gap_samples)
            
            # Pattern: OUTPUT, OUTPUT, INPUT, OUTPUT, INPUT, OUTPUT
            comparison = np.concatenate([
                up_modulated, gap,
                up_modulated, gap,
                wav, gap,
                up_modulated, gap,
                wav, gap,
                up_modulated
            ])
            
            comparison_audio = (fs, (comparison * 32767.0).astype(np.int16))
        
        status = f"✅ Processing complete! Sample rate: {fs}Hz, Duration: {len(wav)/fs:.2f}s"
        
        return output_audio, comparison_audio, status
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


# Create the Gradio interface
with gr.Blocks(title="Sinewave Speech Synthesizer") as demo:
    gr.Markdown("""
    # 🌊 Sinewave Speech Synthesizer
    
    Convert speech into sinewave speech using Linear Predictive Coding (LPC).
    
    **Tips for Best Results:**
    - 🎤 Speak clearly and at a moderate pace
    - 🔇 Record in a quiet environment
    - 🎚️ Start with the default settings, then experiment
    - 👂 For A/B comparison, listen to sinewave *first* before the original!
    
    **What is Sinewave Speech?** Speech represented with just a few frequency-modulated sine waves. 
    Despite extreme simplification, it often remains surprisingly intelligible!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            gr.Markdown("### 🎵 Input Audio")
            audio_input = gr.Audio(
                label="Upload or Record (⚠️ Avoid the scissor/trim tool - it may freeze)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            # Mode selection
            gr.Markdown("### ⚙️ Synthesis Settings")
            mode = gr.Radio(
                choices=["Sinewave", "Buzz", "Noise"],
                value="Sinewave",
                label="Synthesis Mode"
            )
            
            buzz_freq = gr.Slider(
                minimum=40,
                maximum=400,
                value=80,
                step=10,
                label="Buzz Frequency (Hz)",
                visible=False,
                info="Only used in Buzz mode"
            )
            
            # Basic parameters
            gr.Markdown("### Basic Parameters")
            order = gr.Slider(
                minimum=2,
                maximum=15,
                value=4,
                step=1,
                label="Order (number of components)",
                info="More components = more detail but potentially noisier (try 4-5)"
            )
            
            decimation = gr.Slider(
                minimum=1,
                maximum=16,
                value=4,
                step=1,
                label="Decimation Factor",
                info="Higher = faster processing but less detail (try 4-8)"
            )
            
            # Advanced parameters
            with gr.Accordion("Advanced Parameters", open=False):
                low_freq = gr.Slider(
                    minimum=40,
                    maximum=500,
                    value=150,
                    step=10,
                    label="Low Frequency Cutoff (Hz)",
                    info="Lower = more bass (try 100-200 for speech)"
                )
                
                high_freq = gr.Slider(
                    minimum=1000,
                    maximum=5000,
                    value=2800,
                    step=100,
                    label="High Frequency Cutoff (Hz)",
                    info="Critical for quality! Try 2000-3000 for best results"
                )
                
                window_size = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=200,
                    step=10,
                    label="Window Size (samples)",
                    info="Smaller (90-150) = tracks fast speech, Larger (200-300) = smoother"
                )
                
                overlap = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.25,
                    step=0.05,
                    label="Window Overlap",
                    info="Higher = smoother transitions but slower (0.25 is usually good)"
                )
                
                bw_amp = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=60,
                    step=5,
                    label="Bandwidth Amplitude Scaling",
                    info="Higher (60-80) = balanced, Lower (30-40) = emphasize strong formants"
                )
            
            # Comparison options
            gr.Markdown("### Output Options")
            create_comparison = gr.Checkbox(
                label="Generate A/B Comparison",
                value=False,
                info="Creates: output 2x → input → output → input → output"
            )
            
            gap_duration = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.5,
                step=0.1,
                label="Gap Between Segments (seconds)",
                visible=False
            )
            
            # Process button
            process_btn = gr.Button("🎵 Process Audio", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # Outputs
            status_output = gr.Textbox(
                label="Status",
                lines=2,
                interactive=False
            )
            
            audio_output = gr.Audio(
                label="Processed Audio (Sinewave Speech)",
                type="numpy"
            )
            
            comparison_output = gr.Audio(
                label="A/B Comparison",
                type="numpy",
                visible=False
            )
    
    # Show/hide buzz frequency based on mode
    def update_mode(mode_choice):
        return gr.update(visible=(mode_choice == "Buzz"))
    
    mode.change(
        fn=update_mode,
        inputs=[mode],
        outputs=[buzz_freq]
    )
    
    # Show/hide gap duration based on comparison checkbox
    def update_comparison(create_comp):
        return (
            gr.update(visible=create_comp),
            gr.update(visible=create_comp)
        )
    
    create_comparison.change(
        fn=update_comparison,
        inputs=[create_comparison],
        outputs=[gap_duration, comparison_output]
    )
    
    # Process button click
    process_btn.click(
        fn=process_audio,
        inputs=[
            audio_input,
            order,
            low_freq,
            high_freq,
            decimation,
            window_size,
            overlap,
            bw_amp,
            mode,
            buzz_freq,
            create_comparison,
            gap_duration
        ],
        outputs=[audio_output, comparison_output, status_output]
    )
    
    # Examples
    gr.Markdown("### Example Presets")
    gr.Examples(
        examples=[
            ["Sinewave", 4, 150, 2800, 4, 200, 0.25, 60, False, 80],  # Default - good for most speech
            ["Sinewave", 4, 100, 3000, 4, 200, 0.25, 60, True, 80],   # Clear speech with comparison
            ["Sinewave", 5, 200, 3400, 8, 200, 0.25, 60, False, 80],  # More components
            ["Sinewave", 4, 330, 2500, 8, 90, 0.25, 60, False, 80],   # Fast-changing speech
            ["Buzz", 4, 250, 2000, 8, 300, 0.25, 60, False, 80],      # Robotic buzz mode
            ["Noise", 4, 200, 3400, 8, 200, 0.25, 60, False, 80],     # Whisper mode
        ],
        inputs=[
            mode, order, low_freq, high_freq, decimation, 
            window_size, overlap, bw_amp, create_comparison, buzz_freq
        ],
        label="Try these presets (click any row to load settings)"
    )


if __name__ == "__main__":
    demo.launch(share=False)

