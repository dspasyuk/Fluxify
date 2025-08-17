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
import psutil

# Create output folder if it doesn't exist
output_dir = "output"
prompts_dir = "prompts"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(prompts_dir, exist_ok=True)

# Global pipeline variables - Now we only keep one pipeline loaded at a time
current_pipeline = None
current_pipeline_type = None
pipeline_lock = threading.Lock()

# Prompt history file
PROMPTS_FILE = os.path.join(prompts_dir, "prompt_history.json")

def get_memory_usage():
    """Get current memory usage for monitoring"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_cached = torch.cuda.memory_reserved(0)
        return {
            "gpu_total": gpu_memory / 1024**3,
            "gpu_allocated": gpu_allocated / 1024**3, 
            "gpu_cached": gpu_cached / 1024**3,
            "ram_percent": psutil.virtual_memory().percent
        }
    return {"ram_percent": psutil.virtual_memory().percent}

def aggressive_cleanup():
    """More aggressive memory cleanup"""
    global current_pipeline
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        # Clear CUDA cache multiple times
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
    
    # Small delay to allow system to catch up
    time.sleep(0.5)

def unload_current_pipeline():
    """Properly unload the current pipeline"""
    global current_pipeline, current_pipeline_type
    
    if current_pipeline is not None:
        print(f"Unloading {current_pipeline_type} pipeline...")
        
        # Move components to CPU before deletion
        try:
            if hasattr(current_pipeline, 'transformer'):
                current_pipeline.transformer.to('cpu')
            if hasattr(current_pipeline, 'vae'):
                current_pipeline.vae.to('cpu')
            if hasattr(current_pipeline, 'text_encoder'):
                current_pipeline.text_encoder.to('cpu')
            if hasattr(current_pipeline, 'text_encoder_2'):
                current_pipeline.text_encoder_2.to('cpu')
        except Exception as e:
            print(f"Warning: Error moving components to CPU: {e}")
        
        # Delete the pipeline
        del current_pipeline
        current_pipeline = None
        current_pipeline_type = None
        
        # Aggressive cleanup
        aggressive_cleanup()
        
        print("Pipeline unloaded successfully")

def load_pipeline(pipeline_type):
    """Load the specified pipeline type, unloading current one if different"""
    global current_pipeline, current_pipeline_type
    
    if current_pipeline_type == pipeline_type and current_pipeline is not None:
        print(f"{pipeline_type} pipeline already loaded")
        return current_pipeline
    
    # Unload current pipeline if it's different
    if current_pipeline is not None:
        unload_current_pipeline()
    
    # Memory check before loading
    mem_info = get_memory_usage()
    print(f"Memory before loading: RAM {mem_info.get('ram_percent', 0):.1f}%")
    if 'gpu_allocated' in mem_info:
        print(f"GPU: {mem_info['gpu_allocated']:.1f}GB allocated, {mem_info['gpu_cached']:.1f}GB cached")
    
    try:
        print(f"Loading {pipeline_type} pipeline...")
        
        quantization_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True, 
                "bnb_4bit_quant_type": "nf4", 
                "bnb_4bit_compute_dtype": torch.bfloat16
            },
            components_to_quantize=["transformer"],
        )
        
        if pipeline_type == "txt2img":
            current_pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16
            )
        else:  # img2img
            current_pipeline = FluxImg2ImgPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16
            )
        
        # Enable model CPU offload to save VRAM
        current_pipeline.enable_model_cpu_offload()
        current_pipeline_type = pipeline_type
        
        # Memory check after loading
        mem_info = get_memory_usage()
        print(f"Memory after loading: RAM {mem_info.get('ram_percent', 0):.1f}%")
        if 'gpu_allocated' in mem_info:
            print(f"GPU: {mem_info['gpu_allocated']:.1f}GB allocated, {mem_info['gpu_cached']:.1f}GB cached")
        
        print(f"{pipeline_type} pipeline loaded successfully!")
        return current_pipeline
        
    except Exception as e:
        print(f"Error loading {pipeline_type} pipeline: {e}")
        current_pipeline = None
        current_pipeline_type = None
        aggressive_cleanup()
        raise e

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

def generate_images(prompt, image, num_images, guidance_scale, height, width, 
                   num_inference_steps, strength, progress=gr.Progress()):
    """Generate images with proper progress tracking and memory management"""
    if not prompt or prompt.strip() == "":
        return [], "Please enter a prompt"
    
    # Determine pipeline type based on whether image is provided
    pipeline_type = "img2img" if image is not None else "txt2img"
    
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
            # Load appropriate pipeline (this will unload the other if needed)
            progress(0, desc=f"Loading {pipeline_type} pipeline...")
            pipe = load_pipeline(pipeline_type)
            
            generated_images = []
            total_images = int(num_images)
            
            for i in range(total_images):
                try:
                    # Update progress at the start of each image
                    progress((i) / total_images, desc=f"Generating image {i+1}/{total_images} ({pipeline_type})")
                    
                    # Clear memory before each generation (but don't unload pipeline)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Base parameters for both pipelines
                    pipe_kwargs = {
                        "prompt": prompt.strip(),
                        "guidance_scale": float(guidance_scale),
                        "height": int(height),
                        "width": int(width),
                        "num_inference_steps": int(num_inference_steps),
                    }
                    
                    # Add image-specific parameters for img2img
                    if image is not None and pipeline_type == "img2img":
                        pipe_kwargs["image"] = image
                        pipe_kwargs["strength"] = float(strength)
                    
                    # Generate image
                    progress((i + 0.1) / total_images, desc=f"Processing image {i+1}/{total_images}...")
                    with torch.inference_mode():
                        result = pipe(**pipe_kwargs)
                        out = result.images[0]
                    
                    # Update progress after generation
                    progress((i + 0.8) / total_images, desc=f"Saving image {i+1}/{total_images}...")
                    
                    # Save the generated image with metadata
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    filename = f"{output_dir}/{timestamp}_{i:03d}_{pipeline_type}.png"
                    
                    # Save image metadata
                    metadata = {
                        "prompt": prompt.strip(),
                        "settings": settings,
                        "timestamp": timestamp,
                        "filename": os.path.basename(filename),
                        "pipeline_type": pipeline_type
                    }
                    
                    # Save metadata as JSON
                    metadata_file = f"{output_dir}/{timestamp}_{i:03d}_{pipeline_type}_metadata.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    out.save(filename)
                    generated_images.append(filename)
                    
                    # Clear references and do lighter cleanup
                    del result, out
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Final progress update for this image
                    progress((i + 1) / total_images, desc=f"Completed image {i+1}/{total_images}")
                    
                except Exception as e:
                    print(f"Error generating image {i}: {e}")
                    progress((i + 1) / total_images, desc=f"Error on image {i+1}/{total_images}")
                    continue
            
            # Final cleanup and completion
            progress(1.0, desc="Finalizing...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Memory status
            mem_info = get_memory_usage()
            status_msg = f"Generated {len(generated_images)} images successfully using {pipeline_type}! "
            status_msg += f"RAM: {mem_info.get('ram_percent', 0):.1f}%"
            if 'gpu_allocated' in mem_info:
                status_msg += f", GPU: {mem_info['gpu_allocated']:.1f}GB"
            
            return generated_images, status_msg
            
        except Exception as e:
            error_msg = f"Error in generate_images: {e}"
            print(error_msg)
            aggressive_cleanup()
            return [], error_msg

def clear_cache():
    """Manual cache clearing function with pipeline unloading option"""
    aggressive_cleanup()
    mem_info = get_memory_usage()
    status = f"Cache cleared! RAM: {mem_info.get('ram_percent', 0):.1f}%"
    if 'gpu_allocated' in mem_info:
        status += f", GPU: {mem_info['gpu_allocated']:.1f}GB allocated"
    return status

def force_unload_pipeline():
    """Completely unload current pipeline"""
    with pipeline_lock:
        unload_current_pipeline()
        mem_info = get_memory_usage()
        status = f"Pipeline unloaded! RAM: {mem_info.get('ram_percent', 0):.1f}%"
        if 'gpu_allocated' in mem_info:
            status += f", GPU: {mem_info['gpu_allocated']:.1f}GB allocated"
        return status

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

# Preset dimensions
PRESET_SIZES = {
    "Square (512x512)": (512, 512),
    "Square (1024x1024)": (1024, 1024),
    "Portrait (1472x832)": (1472, 832),
    "Landscape (832x1472)": (832, 1472),
    "Widescreen (1080x1920)": (1080, 1920),
    "Custom": None
}

def update_dimensions(preset):
    """Update width/height based on preset selection"""
    if preset in PRESET_SIZES and PRESET_SIZES[preset]:
        height, width = PRESET_SIZES[preset]
        return (
            gr.update(value=height, interactive=preset=="Custom"),
            gr.update(value=width, interactive=preset=="Custom")
        )
    return (
        gr.update(interactive=True),
        gr.update(interactive=True)
    )

# Create Gradio interface
with gr.Blocks(title="Fluxify", theme=gr.themes.Default(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink)) as demo:
    gr.Markdown("# 🎨 Fluxify")
    gr.Markdown("Generate high-quality images using FLUX.1-dev model with improved memory management")
    
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
                    value="Portrait (1472x832)",
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
            
            with gr.Row():
                unload_button = gr.Button("🔄 Unload Pipeline", variant="secondary")
        
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
                lines=3
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
        outputs=[height, width]
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
    
    # Unload pipeline
    unload_button.click(
        fn=force_unload_pipeline,
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
        with pipeline_lock:
            unload_current_pipeline()
            aggressive_cleanup()
