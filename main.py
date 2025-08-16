import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from diffusers.quantizers import PipelineQuantizationConfig
import os
from datetime import datetime
from PIL import Image
import gradio as gr
import gc
import threading
import time
import json

# Create output folder if it doesn't exist
output_dir = "output"
prompts_dir = "prompts"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(prompts_dir, exist_ok=True)

# Global pipeline variables
pipe = None
img2img_pipe = None
pipeline_lock = threading.Lock()

# Prompt history file
PROMPTS_FILE = os.path.join(prompts_dir, "prompt_history.json")

def load_prompt_history():
    """Load prompt history from JSON file"""
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_prompt_to_history(prompt, settings):
    """Save prompt and settings to history"""
    if not prompt.strip():
        return
    
    history = load_prompt_history()
    
    # Create new entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt.strip(),
        "settings": settings
    }
    
    # Avoid duplicates - check if same prompt exists
    for existing in history:
        if existing.get("prompt") == prompt.strip():
            return  # Don't add duplicate
    
    # Add to beginning of list and limit to 100 entries
    history.insert(0, entry)
    history = history[:100]
    
    # Save back to file
    try:
        with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving prompt history: {e}")

def get_prompt_choices():
    """Get list of recent prompts for dropdown"""
    history = load_prompt_history()
    choices = [""] + [entry["prompt"][:100] + "..." if len(entry["prompt"]) > 100 else entry["prompt"] 
                     for entry in history[:20]]  # Show only recent 20
    return choices

def load_prompt_settings(selected_prompt):
    """Load settings for selected prompt"""
    if not selected_prompt or selected_prompt == "":
        return "", 3.5, 832, 1472, 10, 0.8
    
    history = load_prompt_history()
    for entry in history:
        prompt_preview = entry["prompt"][:100] + "..." if len(entry["prompt"]) > 100 else entry["prompt"]
        if prompt_preview == selected_prompt:
            settings = entry.get("settings", {})
            return (
                entry["prompt"],
                settings.get("guidance_scale", 3.5),
                settings.get("height", 832),
                settings.get("width", 1472),
                settings.get("num_inference_steps", 10),
                settings.get("strength", 0.8)
            )
    return selected_prompt, 3.5, 832, 1472, 10, 0.8

def initialize_pipeline():
    """Initialize the FLUX pipeline with proper error handling"""
    global pipe
    if pipe is None:
        try:
            print("Loading FLUX text-to-image pipeline...")
            quantization_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True, 
                    "bnb_4bit_quant_type": "nf4", 
                    "bnb_4bit_compute_dtype": torch.bfloat16
                },
                components_to_quantize=["transformer"],
            )
            
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16
            )
            
            # Enable model CPU offload to save VRAM
            pipe.enable_model_cpu_offload()
            print("Text-to-image pipeline loaded successfully!")
            
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise e

def initialize_img2img_pipeline():
    """Initialize the FLUX img2img pipeline"""
    global img2img_pipe
    if img2img_pipe is None:
        try:
            print("Loading FLUX image-to-image pipeline...")
            quantization_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True, 
                    "bnb_4bit_quant_type": "nf4", 
                    "bnb_4bit_compute_dtype": torch.bfloat16
                },
                components_to_quantize=["transformer"],
            )
            
            img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16
            )
            
            # Enable model CPU offload to save VRAM
            img2img_pipe.enable_model_cpu_offload()
            print("Image-to-image pipeline loaded successfully!")
            
        except Exception as e:
            print(f"Error loading img2img pipeline: {e}")
            raise e

def cleanup_memory():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    time.sleep(0.1)  # Small delay to allow cleanup

def generate_images(prompt, image, num_images, guidance_scale, height, width, 
                   num_inference_steps, strength, progress=gr.Progress()):
    """Generate images with proper progress tracking and memory management"""
    if not prompt or prompt.strip() == "":
        return [], "Please enter a prompt"
    
    # Save prompt and settings to history
    settings = {
        "guidance_scale": guidance_scale,
        "height": int(height),
        "width": int(width),
        "num_inference_steps": int(num_inference_steps),
        "strength": strength,
        "has_input_image": image is not None
    }
    save_prompt_to_history(prompt, settings)
    
    with pipeline_lock:  # Ensure thread safety
        try:
            # Initialize appropriate pipeline
            if image is not None:
                initialize_img2img_pipeline()
                current_pipe = img2img_pipe
                pipe_type = "img2img"
            else:
                initialize_pipeline()
                current_pipe = pipe
                pipe_type = "txt2img"
            
            generated_images = []
            
            for i in progress.tqdm(range(int(num_images)), desc=f"Generating images ({pipe_type})"):
                try:
                    # Clear memory before each generation
                    cleanup_memory()
                    
                    # Base parameters for both pipelines
                    pipe_kwargs = {
                        "prompt": prompt.strip(),
                        "guidance_scale": float(guidance_scale),
                        "height": int(height),
                        "width": int(width),
                        "num_inference_steps": int(num_inference_steps),
                    }
                    
                    # Add image-specific parameters for img2img
                    if image is not None and pipe_type == "img2img":
                        pipe_kwargs["image"] = image
                        pipe_kwargs["strength"] = float(strength)
                    
                    # Generate image
                    with torch.inference_mode():
                        result = current_pipe(**pipe_kwargs)
                        out = result.images[0]
                    
                    # Save the generated image with metadata
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    filename = f"{output_dir}/{timestamp}_{i:03d}_{pipe_type}.png"
                    
                    # Save image metadata
                    metadata = {
                        "prompt": prompt.strip(),
                        "settings": settings,
                        "timestamp": timestamp,
                        "filename": os.path.basename(filename),
                        "pipeline_type": pipe_type
                    }
                    
                    # Save metadata as JSON
                    metadata_file = f"{output_dir}/{timestamp}_{i:03d}_{pipe_type}_metadata.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    out.save(filename)
                    generated_images.append(filename)
                    
                    # Clear references
                    del result, out
                    cleanup_memory()
                    
                except Exception as e:
                    print(f"Error generating image {i}: {e}")
                    continue
            
            # Final cleanup
            cleanup_memory()
            
            status_msg = f"Generated {len(generated_images)} images successfully using {pipe_type}!"
            return generated_images, status_msg
            
        except Exception as e:
            error_msg = f"Error in generate_images: {e}"
            print(error_msg)
            cleanup_memory()
            return [], error_msg

def clear_cache():
    """Manual cache clearing function"""
    cleanup_memory()
    return "Cache cleared!"

def export_prompts():
    """Export all prompts to a downloadable JSON file"""
    history = load_prompt_history()
    export_file = os.path.join(prompts_dir, f"prompts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    try:
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return f"Prompts exported to {export_file}", export_file
    except Exception as e:
        return f"Error exporting prompts: {e}", None

# Initialize pipelines on startup
try:
    initialize_pipeline()  # Load text-to-image by default
except Exception as e:
    print(f"Failed to initialize pipeline: {e}")

# Preset dimensions
PRESET_SIZES = {
    "Square (1024x1024)": (1024, 1024),
    "Portrait (832x1472)": (832, 1472),
    "Landscape (1472x832)": (1472, 832),
    "Widescreen (1920x1080)": (1920, 1080),
    "Custom": None
}

def update_dimensions(preset):
    """Update width/height based on preset selection"""
    if preset in PRESET_SIZES and PRESET_SIZES[preset]:
        width, height = PRESET_SIZES[preset]
        return gr.update(value=height), gr.update(value=width), gr.update(interactive=preset=="Custom"), gr.update(interactive=preset=="Custom")
    return gr.update(), gr.update(), gr.update(interactive=True), gr.update(interactive=True)

# Create Gradio interface
with gr.Blocks(title="Fluxify", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Fluxify")
    gr.Markdown("Generate high-quality images using FLUX.1-dev model with advanced controls")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image input
            image = gr.Image(
                type="pil", 
                label="Input Image (optional for img2img)"
            )
            
            # Generation parameters
            with gr.Group():
                gr.Markdown("### ⚙️ Generation Parameters")
                
                num_images = gr.Slider(
                    minimum=1, 
                    maximum=5,
                    value=1, 
                    step=1, 
                    label="Number of Images"
                )
                
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=3.5,
                    step=0.1,
                    label="Guidance Scale (creativity vs prompt following)"
                )
                
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Inference Steps (quality vs speed)"
                )
                
                strength = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.8,
                    step=0.05,
                    label="Img2Img Strength (when input image provided)"
                )
            
            # Dimensions
            with gr.Group():
                gr.Markdown("### 📐 Image Dimensions")
                
                size_preset = gr.Dropdown(
                    choices=list(PRESET_SIZES.keys()),
                    value="Portrait (832x1472)",
                    label="Size Preset"
                )
                
                with gr.Row():
                    height = gr.Number(
                        value=832,
                        label="Height",
                        minimum=512,
                        maximum=2048,
                        step=8
                    )
                    width = gr.Number(
                        value=1472,
                        label="Width", 
                        minimum=512,
                        maximum=2048,
                        step=8
                    )
            
            # Action buttons
            with gr.Row():
                generate_button = gr.Button("🚀 Generate", variant="primary", scale=2)
                clear_button = gr.Button("🧹 Clear Cache", variant="secondary", scale=1)
        
        with gr.Column(scale=2):
            # Prompt section (now at top of right column)
            with gr.Group():
                gr.Markdown("### 📝 Prompt")
                
                prompt_history = gr.Dropdown(
                    choices=get_prompt_choices(),
                    label="Recent Prompts",
                    value="",
                    interactive=True
                )
                
                prompt = gr.Textbox(
                    label="Prompt", 
                    lines=4, 
                    placeholder="Enter your detailed prompt here...",
                    value=""
                )
                
                with gr.Row():
                    refresh_prompts = gr.Button("🔄 Refresh", size="sm")
                    export_btn = gr.Button("📥 Export Prompts", size="sm")
            
            # Output section
            output_gallery = gr.Gallery(
                label="Generated Images",
                columns=2,
                rows=2,
                height=600,
                show_label=True
            )
            
            status_output = gr.Textbox(
                label="Status", 
                interactive=False,
                lines=2
            )
            
            export_status = gr.Textbox(
                label="Export Status",
                interactive=False,
                visible=False
            )
            
            download_file = gr.File(
                label="Download Exported Prompts",
                visible=False
            )
    
    # Event handlers
    
    # Update dimensions when preset changes
    size_preset.change(
        fn=update_dimensions,
        inputs=[size_preset],
        outputs=[height, width, height, width]
    )
    
    # Load prompt when selected from history
    prompt_history.change(
        fn=load_prompt_settings,
        inputs=[prompt_history],
        outputs=[prompt, guidance_scale, height, width, num_inference_steps, strength]
    )
    
    # Refresh prompt choices
    refresh_prompts.click(
        fn=lambda: gr.update(choices=get_prompt_choices()),
        outputs=[prompt_history]
    )
    
    # Main generation
    generate_button.click(
        fn=generate_images,
        inputs=[prompt, image, num_images, guidance_scale, height, width, num_inference_steps, strength],
        outputs=[output_gallery, status_output],
        show_progress=True
    )
    
    # Clear cache
    clear_button.click(
        fn=clear_cache,
        outputs=status_output
    )
    
    # Export prompts
    export_btn.click(
        fn=export_prompts,
        outputs=[export_status, download_file]
    ).then(
        lambda: (gr.update(visible=True), gr.update(visible=True)),
        outputs=[export_status, download_file]
    )

if __name__ == "__main__":
    try:
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            quiet=False
        )
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Cleanup on exit
        cleanup_memory()
        if pipe is not None:
            del pipe
        if img2img_pipe is not None:
            del img2img_pipe