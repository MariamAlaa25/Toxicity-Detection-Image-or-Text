from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize device and BLIP model only once
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)
model.eval()

# max_length:the generated caption max len
def generate_caption(image: Image.Image, max_length: int = 50) -> str:
    # Convert image to RGB 
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=max_length)
    
    # Decode caption
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption
