# 🎨 Fluxify

A simple Gradio web interface for FLUX.1-dev image generation with text-to-image and image-to-image support.

## Features
- **Text-to-Image & Image-to-Image** generation
- **Advanced controls** (guidance scale, steps, dimensions)
- **Prompt memory** with automatic saving/loading
- **Memory optimized** with 4-bit quantization
- **Preset dimensions** (Square, Portrait, Landscape, etc.)

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Open `http://localhost:7860` in your browser or use provided public url.

## Usage
1. **Text-to-Image**: Enter prompt → Generate
2. **Image-to-Image**: Upload image + prompt → Generate  
3. **Recall prompts**: Use dropdown to reload previous settings

## Requirements
- Python 3.8+
- CUDA GPU recommended (works on CPU) 
- ~30GB disk space for model and about 16 Gb of VRAM 


<img width="1169" height="878" alt="image" src="https://github.com/user-attachments/assets/4361ddd7-e387-43bf-9175-ec60e529984b" />

