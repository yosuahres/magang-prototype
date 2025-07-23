import torch
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# --- Configuration ---
# URL of the image to perform semantic segmentation on
IMAGE_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'

# Pre-trained model for semantic segmentation.
# SegFormer is a good choice for this task.
# "nvidia/segformer-b0-finetuned-ade-512-512" is a small, efficient model
# "nvidia/segformer-b5-finetuned-ade-640-640" is a larger, more accurate model
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

# --- Step 1: Load the image ---
print(f"Loading image from: {IMAGE_URL}")
try:
    image = Image.open(requests.get(IMAGE_URL, stream=True).raw).convert("RGB")
    print(f"Original image size: {image.width}x{image.height}")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# --- Step 2: Load the image processor and the semantic segmentation model ---
print(f"Loading image processor and model: {MODEL_NAME}")
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded and moved to: {device}")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit()

# --- Step 3: Preprocess the image and prepare inputs for the model ---
# The processor resizes the image and normalizes pixel values
print("Preprocessing image...")
inputs = processor(images=image, return_tensors="pt")
# Move inputs to the same device as the model
inputs = {k: v.to(device) for k, v in inputs.items()}
print(f"Processed input tensor shape: {inputs['pixel_values'].shape}")

# --- Step 4: Perform inference ---
print("Performing semantic segmentation inference...")
try:
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(**inputs)
    print("Inference complete.")
except Exception as e:
    print(f"Error during inference: {e}")
    exit()

# --- Step 5: Process the model output to get the segmentation map ---
# The model outputs logits, which need to be upsampled to the original image size
# and then converted into a segmentation map (pixel-wise class predictions).
logits = outputs.logits.cpu() # Move logits to CPU for further processing
print(f"Raw logits shape: {logits.shape}")

# Upsample the logits to the original image dimensions
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1], # Target size is (height, width)
    mode="bilinear",
    align_corners=False,
)
print(f"Upsampled logits shape: {upsampled_logits.shape}")

# Get the predicted segmentation map by taking the argmax along the channel dimension
# The result is a 2D tensor where each element is the predicted class ID for that pixel.
segmentation_map = upsampled_logits.argmax(dim=1)[0]
print(f"Segmentation map shape: {segmentation_map.shape}")

# --- Step 6: Visualize the segmentation result ---
print("Visualizing segmentation result...")

# Define a more visually appealing and consistent colormap
# This colormap is a list of RGB tuples (0-255 values).
# You can expand this list with more colors if you expect many unique classes.
COLORMAP = [
    (0, 0, 0),         # Black (for background or unknown)
    (128, 0, 0),       # Dark Red
    (0, 128, 0),       # Dark Green
    (128, 128, 0),     # Dark Yellow
    (0, 0, 128),       # Dark Blue
    (128, 0, 128),     # Dark Magenta
    (0, 128, 128),     # Dark Cyan
    (128, 128, 128),   # Grey
    (64, 0, 0),        # Brown
    (192, 0, 0),       # Red
    (64, 128, 0),      # Olive Green
    (192, 128, 0),     # Orange
    (64, 0, 128),      # Purple
    (192, 0, 128),     # Pink
    (64, 128, 128),    # Teal
    (192, 128, 128),   # Light Grey
    (0, 64, 0),        # Forest Green
    (128, 64, 0),      # Dark Orange
    (0, 192, 0),       # Bright Green
    (128, 192, 0),     # Lime Green
    (0, 64, 128),      # Dark Teal
    (128, 64, 128),    # Plum
    (0, 192, 128),     # Aqua
    (128, 192, 128),   # Mint Green
    (64, 64, 0),       # Dark Olive
    (192, 64, 0),      # Rust
    (64, 192, 0),      # Light Green
    (192, 192, 0),     # Gold
    (64, 64, 128),     # Slate Blue
    (192, 64, 128),    # Orchid
    (64, 192, 128),    # Turquoise
    (192, 192, 128),   # Pale Yellow
]

# Create a dictionary to store colors for each unique class ID found in the map
unique_classes = torch.unique(segmentation_map).tolist()
class_colors = {}
for i, class_id in enumerate(unique_classes):
    # Cycle through the predefined COLORMAP
    class_colors[class_id] = COLORMAP[i % len(COLORMAP)]

# Create an empty RGB image for the colored segmentation mask
colored_segmentation_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)

# Populate the colored mask based on the segmentation map
for class_id in unique_classes:
    # Get pixels belonging to the current class
    mask_for_class = (segmentation_map == class_id).cpu().numpy()
    # Apply the assigned color to these pixels
    colored_segmentation_mask[mask_for_class] = class_colors[class_id]

# Convert the original image and the colored mask to numpy arrays for plotting
original_image_np = np.array(image)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Plot the original image
axes[0].imshow(original_image_np)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Plot the colored segmentation mask
axes[1].imshow(colored_segmentation_mask)
axes[1].set_title("Semantic Segmentation Mask")
axes[1].axis('off')

# Add a legend for the classes if id2label is available
if hasattr(model.config, 'id2label'):
    labels = [model.config.id2label[class_id] for class_id in unique_classes]
    # Ensure colors in legend match the actual colors used
    handles = [plt.Rectangle((0,0),1,1, color=np.array(class_colors[class_id])/255) for class_id in unique_classes]
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5))

plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.show()

print("Program finished.")
