import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

class DINOv2SemanticSegmentation:
    def __init__(self, model_name="dinov2_vits14", device=None):
        """
        Initialize DINOv2 semantic segmentation model
        
        Args:
            model_name: DINOv2 model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
            device: Computing device (cuda/cpu)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load DINOv2 model
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),  # DINOv2 standard size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Define semantic classes (you can modify these based on your needs)
        self.class_names = [
            'glass',
        ]
        
        # Initialize classifier components
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_classifier_trained = False
        
    def extract_features(self, image_path):
        """
        Extract dense features from an image using DINOv2
        
        Args:
            image_path: Path to input image
            
        Returns:
            features: Dense feature map
            original_image: Original PIL image
        """
        # Load and preprocess image
        original_image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract features
            features = self.model.forward_features(image_tensor)
            
            # Get patch tokens (excluding CLS token)
            patch_tokens = features['x_norm_patchtokens']
            
            # Reshape to spatial dimensions
            # DINOv2 uses 14x14 patches for 518x518 input
            B, N, D = patch_tokens.shape
            H = W = int(N ** 0.5)  # 37x37 for 518x518 input
            features_spatial = patch_tokens.reshape(B, H, W, D)
            
        return features_spatial.squeeze(0), original_image
    
    def cluster_features(self, features, n_clusters=8):
        """
        Cluster features to create semantic segments
        
        Args:
            features: Dense feature map (H, W, D)
            n_clusters: Number of semantic clusters
            
        Returns:
            segmentation_map: Cluster assignments for each pixel
        """
        H, W, D = features.shape
        features_flat = features.reshape(-1, D).cpu().numpy()
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_flat)
        
        # Reshape back to spatial dimensions
        segmentation_map = cluster_labels.reshape(H, W)
        
        return segmentation_map
    
    def create_simple_classifier(self, features, segmentation_map):
        """
        Create a simple rule-based classifier for semantic classes
        
        Args:
            features: Dense feature map (H, W, D)
            segmentation_map: Cluster assignments
            
        Returns:
            class_predictions: Predicted class for each cluster
        """
        H, W, D = features.shape
        features_flat = features.reshape(-1, D).cpu().numpy()
        
        # Calculate cluster centers and properties
        n_clusters = len(np.unique(segmentation_map))
        cluster_features = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = (segmentation_map == cluster_id)
            if np.sum(cluster_mask) == 0:
                continue
                
            # Get features for this cluster
            cluster_pixels = features_flat[cluster_mask.flatten()]
            
            # Calculate statistics
            mean_features = np.mean(cluster_pixels, axis=0)
            std_features = np.std(cluster_pixels, axis=0)
            cluster_size = np.sum(cluster_mask) / (H * W)  # Relative size
            
            # Position analysis (top regions likely sky, bottom likely road)
            y_coords, x_coords = np.where(cluster_mask)
            avg_y = np.mean(y_coords) / H  # Normalized average Y position
            avg_x = np.mean(x_coords) / W  # Normalized average X position
            
            cluster_features.append({
                'id': cluster_id,
                'mean_features': mean_features,
                'std_features': std_features,
                'size': cluster_size,
                'avg_y': avg_y,
                'avg_x': avg_x,
                'pixel_count': np.sum(cluster_mask)
            })
        
        # Simple rule-based classification
        class_predictions = {}
        
        # Sort clusters by size (largest first)
        sorted_clusters = sorted(cluster_features, key=lambda x: x['size'], reverse=True)
        
        for i, cluster_info in enumerate(sorted_clusters):
            cluster_id = cluster_info['id']
            avg_y = cluster_info['avg_y']
            size = cluster_info['size']
            
            # Rule-based classification
            if avg_y < 0.3:  # Top region
                if size > 0.1:  # Large cluster at top
                    predicted_class = 'sky'
                else:
                    predicted_class = 'building'
            elif avg_y > 0.7:  # Bottom region
                if size > 0.05:  # Reasonable size at bottom
                    predicted_class = 'road'
                else:
                    predicted_class = 'object'
            else:  # Middle region
                if size > 0.2:  # Large middle cluster
                    predicted_class = 'vegetation'
                elif size > 0.05:
                    predicted_class = 'building'
                elif size > 0.01:
                    predicted_class = 'car'
                else:
                    predicted_class = 'person'
            
            # Assign the largest unassigned cluster as background if needed
            if i == 0 and size > 0.3:
                predicted_class = 'background'
                
            class_predictions[cluster_id] = predicted_class
        
        return class_predictions
    
    def predict_semantic_classes(self, features, segmentation_map):
        """
        Predict semantic classes for each segment
        
        Args:
            features: Dense feature map (H, W, D)
            segmentation_map: Cluster assignments
            
        Returns:
            class_map: Map with predicted class for each pixel
            class_predictions: Dictionary mapping cluster_id to class_name
        """
        # Get class predictions for each cluster
        class_predictions = self.create_simple_classifier(features, segmentation_map)
        
        # Create class map
        class_map = np.zeros_like(segmentation_map, dtype=object)
        
        for cluster_id, class_name in class_predictions.items():
            mask = (segmentation_map == cluster_id)
            class_map[mask] = class_name
        
        return class_map, class_predictions
    
    def create_class_colored_segmentation(self, class_map, class_predictions):
        """
        Create a colored visualization based on predicted classes
        
        Args:
            class_map: Map with predicted class for each pixel
            class_predictions: Dictionary mapping cluster_id to class_name
            
        Returns:
            colored_segmentation: RGB colored segmentation based on classes
        """
        # Define colors for each class
        class_colors = {
            'sky': [135, 206, 235],      # Sky blue
            'building': [169, 169, 169],  # Gray
            'road': [64, 64, 64],        # Dark gray
            'tree': [34, 139, 34],       # Forest green
            'vegetation': [124, 252, 0],  # Lawn green
            'water': [0, 191, 255],      # Deep sky blue
            'person': [255, 20, 147],    # Deep pink
            'car': [255, 0, 0],          # Red
            'object': [255, 165, 0],     # Orange
            'background': [0, 0, 0]      # Black
        }
        
        H, W = class_map.shape
        colored_segmentation = np.zeros((H, W, 3), dtype=np.uint8)
        
        for class_name, color in class_colors.items():
            mask = (class_map == class_name)
            colored_segmentation[mask] = color
        
        return colored_segmentation
    
    def resize_segmentation(self, segmentation_map, target_size):
        """
        Resize segmentation map to match original image size
        
        Args:
            segmentation_map: Cluster assignments
            target_size: Target size (width, height)
            
        Returns:
            resized_segmentation: Resized segmentation map
        """
        resized_segmentation = cv2.resize(
            segmentation_map.astype(np.uint8), 
            target_size, 
            interpolation=cv2.INTER_NEAREST
        )
        return resized_segmentation
    
    def create_colored_segmentation(self, segmentation_map, n_clusters=8):
        """
        Create a colored visualization of the segmentation
        
        Args:
            segmentation_map: Cluster assignments
            n_clusters: Number of clusters
            
        Returns:
            colored_segmentation: RGB colored segmentation
        """
        # Create color palette - bright colors for foreground objects
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))[:, :3]  # RGB only
        
        # Make the most common cluster (likely background) black
        unique, counts = np.unique(segmentation_map, return_counts=True)
        background_cluster = unique[np.argmax(counts)]
        
        H, W = segmentation_map.shape
        colored_segmentation = np.zeros((H, W, 3))
        
        for i in range(n_clusters):
            mask = segmentation_map == i
            if i == background_cluster:
                # Set background to black
                colored_segmentation[mask] = [0, 0, 0]
            else:
                # Use bright colors for foreground objects
                colored_segmentation[mask] = colors[i]
        
        return (colored_segmentation * 255).astype(np.uint8)
    
    def segment_image(self, image_path, n_clusters=8, predict_classes=True, save_results=True):
        """
        Complete semantic segmentation pipeline
        
        Args:
            image_path: Path to input image
            n_clusters: Number of semantic segments
            predict_classes: Whether to predict semantic classes
            save_results: Whether to save visualization results
            
        Returns:
            segmentation_map: Final segmentation map
            colored_segmentation: Colored visualization
            class_map: Predicted classes map (if predict_classes=True)
            class_predictions: Class predictions dictionary (if predict_classes=True)
        """
        print(f"Processing image: {image_path}")
        
        # Extract features
        features, original_image = self.extract_features(image_path)
        print(f"Extracted features shape: {features.shape}")
        
        # Cluster features
        segmentation_map = self.cluster_features(features, n_clusters)
        print(f"Created {n_clusters} semantic segments")
        
        # Resize to original image size
        original_size = original_image.size  # (width, height)
        segmentation_resized = self.resize_segmentation(segmentation_map, original_size)
        
        results = {
            'segmentation_map': segmentation_resized,
            'colored_segmentation': None,
            'class_map': None,
            'class_predictions': None
        }
        
        if predict_classes:
            # Predict semantic classes
            class_map, class_predictions = self.predict_semantic_classes(features, segmentation_map)
            print(f"Predicted classes: {list(set(class_predictions.values()))}")
            
            # Create class-based colored segmentation first (before resizing)
            colored_segmentation = self.create_class_colored_segmentation(class_map, class_predictions)
            
            # Resize the colored segmentation instead of the class map
            colored_segmentation = cv2.resize(
                colored_segmentation, 
                original_size, 
                interpolation=cv2.INTER_NEAREST
            )
            
            # For the class map, we'll create a numeric version that can be resized
            class_to_id = {class_name: idx for idx, class_name in enumerate(set(class_predictions.values()))}
            numeric_class_map = np.zeros_like(segmentation_map, dtype=np.uint8)
            
            for cluster_id, class_name in class_predictions.items():
                mask = (segmentation_map == cluster_id)
                numeric_class_map[mask] = class_to_id[class_name]
            
            # Resize the numeric class map
            class_map_resized = cv2.resize(
                numeric_class_map, 
                original_size, 
                interpolation=cv2.INTER_NEAREST
            )
            
            results['class_map'] = class_map_resized
            results['class_predictions'] = class_predictions
        else:
            # Create regular colored visualization
            colored_segmentation = self.create_colored_segmentation(segmentation_resized, n_clusters)
        
        results['colored_segmentation'] = colored_segmentation
        
        if save_results:
            self.visualize_results(original_image, segmentation_resized, colored_segmentation, 
                                 image_path, class_predictions if predict_classes else None)
        
        return results
    
    def visualize_results(self, original_image, segmentation_map, colored_segmentation, image_path, class_predictions=None):
        """
        Visualize segmentation results
        
        Args:
            original_image: Original PIL image
            segmentation_map: Segmentation map
            colored_segmentation: Colored segmentation
            image_path: Original image path for naming
            class_predictions: Dictionary of class predictions (optional)
        """
        # Set black background style
        plt.style.use('dark_background')
        
        if class_predictions:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        fig.patch.set_facecolor('black')
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", color='white')
        axes[0].axis('off')
        axes[0].set_facecolor('black')
        
        # Segmentation map
        axes[1].imshow(segmentation_map, cmap='tab10')
        axes[1].set_title("Segmentation Map", color='white')
        axes[1].axis('off')
        axes[1].set_facecolor('black')
        
        # Colored segmentation
        axes[2].imshow(colored_segmentation)
        if class_predictions:
            axes[2].set_title("Class-based Segmentation", color='white')
        else:
            axes[2].set_title("Colored Segmentation", color='white')
        axes[2].axis('off')
        axes[2].set_facecolor('black')
        
        # If we have class predictions, show them
        if class_predictions:
            # Create text summary of predictions
            class_text = "Predicted Classes:\n\n"
            for cluster_id, class_name in class_predictions.items():
                class_text += f"Segment {cluster_id}: {class_name}\n"
            
            axes[3].text(0.1, 0.9, class_text, transform=axes[3].transAxes, 
                        fontsize=10, verticalalignment='top', color='white')
            axes[3].set_title("Class Predictions", color='white')
            axes[3].axis('off')
            axes[3].set_facecolor('black')
        
        plt.tight_layout()
        
        # Save visualization with black background
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if class_predictions:
            output_path = f"{base_name}_class_segmentation.png"
        else:
            output_path = f"{base_name}_segmentation.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Saved visualization: {output_path}")
        
        plt.show()
        
        # Reset style to default
        plt.style.use('default')
    
    def create_overlay(self, original_image, colored_segmentation, alpha=0.6):
        """
        Create overlay of original image with segmentation
        
        Args:
            original_image: Original PIL image
            colored_segmentation: Colored segmentation
            alpha: Transparency factor
            
        Returns:
            overlay: Blended image
        """
        original_array = np.array(original_image)
        overlay = cv2.addWeighted(original_array, 1-alpha, colored_segmentation, alpha, 0)
        return overlay

def main():
    """
    Main function to run semantic segmentation
    """
    # Initialize the segmentation model
    segmenter = DINOv2SemanticSegmentation(model_name="dinov2_vits14")
    
    # Check for images in the images directory
    images_dir = "images"
    if not os.path.exists(images_dir):
        print(f"Images directory '{images_dir}' not found. Please ensure it exists with images.")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(images_dir, file))
    
    if not image_files:
        print(f"No image files found in '{images_dir}' directory.")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in image_files:
        try:
            # Perform semantic segmentation with class prediction
            results = segmenter.segment_image(
                image_path, 
                n_clusters=8,  # Adjust number of segments as needed
                predict_classes=True,  # Enable class prediction
                save_results=True
            )
            
            # Extract results
            segmentation_map = results['segmentation_map']
            colored_segmentation = results['colored_segmentation']
            class_map = results['class_map']
            class_predictions = results['class_predictions']
            
            # Print class predictions
            if class_predictions:
                print(f"Class predictions for {os.path.basename(image_path)}:")
                for cluster_id, class_name in class_predictions.items():
                    print(f"  Segment {cluster_id}: {class_name}")
            
            # Create overlay visualization
            original_image = Image.open(image_path).convert("RGB")
            overlay = segmenter.create_overlay(original_image, colored_segmentation)
            
            # Save overlay
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            overlay_path = f"{base_name}_overlay.png"
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"Saved overlay: {overlay_path}")
            
            print(f"Completed processing: {image_path}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
