import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import psutil
from transformers import ViTImageProcessor, ViTModel

# Configuration parameters
HIDDEN_SIZE = 64
FRAMES_PER_EPISODE = 5  # Number of frames to sample per episode
MAX_EPISODES = 4  # Number of episodes to process per game
MAX_GAMES = 20  # Maximum number of games to process
BATCH_SIZE = 1  # Process frames one by one

def print_memory_stats():
    """Print memory usage statistics"""
    process = psutil.Process(os.getpid())
    print(f"CPU Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB (allocated)")
        print(f"GPU Cache: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB (reserved)")

def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def init_vit_model():
    """Initialize and cache ViT model"""
    model_path = "vit-base-patch16-224"
    try:
        print("Loading ViT model from local cache...")
        # Load model on CPU first
        vit_model = ViTModel.from_pretrained(model_path, local_files_only=True)
        vit_processor = ViTImageProcessor.from_pretrained(model_path, local_files_only=True)
        
        # Check GPU memory
        print_memory_stats()
        if torch.cuda.is_available():
            # Use float16 to reduce memory usage
            vit_model = vit_model.half().cuda()
            
        print("Model loading completed")
        print_memory_stats()
        
    except Exception as e:
        print(f"Loading failed: {e}")
        raise
    
    return vit_model, vit_processor

def load_and_process_frame(frame_data):
    """Load and process a single frame
    
    Args:
        frame_data: Raw frame data from episodes
    Returns:
        PIL.Image: Processed RGB image or None if processing fails
    """
    try:
        frame = np.array(frame_data)
        
        if frame is None or np.all(frame == 0):
            return None
            
        # Normalize frame values
        if frame.dtype in [np.float64, np.float32]:
            if frame.max() <= 0:
                return None
            frame = (frame * 255 / frame.max()).astype(np.uint8)
            
        # Convert to RGB
        if frame.shape[-1] == 4:
            frame_gray = frame[:, :, -1]
            if np.all(frame_gray == 0):
                return None
            frame_rgb = np.stack([frame_gray] * 3, axis=-1)
            return Image.fromarray(frame_rgb)
            
    except Exception as e:
        print(f"Frame processing failed: {e}")
    return None

def extract_features(vit_model, vit_processor, frame, feature_reducer):
    """Extract features from a single frame using ViT model
    
    Args:
        vit_model: ViT model instance
        vit_processor: ViT image processor
        frame: PIL Image
        feature_reducer: Linear layer for dimensionality reduction
    Returns:
        numpy.ndarray: Extracted features or None if extraction fails
    """
    if frame is None:
        return None
        
    try:
        with torch.no_grad():
            # Process image on CPU
            inputs = vit_processor(images=frame, return_tensors="pt")
            
            # Move to GPU
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            # Use half precision
            with torch.cuda.amp.autocast():
                outputs = vit_model(**inputs)
                features = outputs.last_hidden_state[:, 0]
                features = feature_reducer(features)
            
            # Cleanup immediately
            del inputs, outputs
            clear_gpu_memory()
            
            return features.cpu().numpy()
            
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

def main():
    try:
        print("Initializing...")
        print_memory_stats()
        
        # Initialize model
        vit_model, vit_processor = init_vit_model()
        vit_feature_reducer = nn.Linear(768, HIDDEN_SIZE).half().cuda()
        
        # Get game folders
        base_path = "/home/llm_user/yxh/policy-diffusion/param_data/Atari_zoo"
        game_folders = sorted([f for f in glob.glob(os.path.join(base_path, "Atari-*"))
                       if os.path.isdir(f)])[:MAX_GAMES]  # Sort for consistent order
        
        print(f"Processing {len(game_folders)} games")
        all_features = []
        game_labels = []
        
        # Process each game
        for game_folder in tqdm(game_folders, desc="Processing games"):
            game_name = os.path.basename(game_folder)
            data_path = os.path.join(game_folder, "200", "episodes_data2.pkl")
            
            if not os.path.exists(data_path):
                continue
                
            try:
                # Load and process frames
                with open(data_path, 'rb') as f:
                    episodes_data = pickle.load(f)
                
                if not episodes_data or not episodes_data[0]:
                    continue
                    
                # Process multiple episodes
                for episode_idx in range(min(MAX_EPISODES, len(episodes_data[0]))):
                    episode = episodes_data[0][episode_idx]
                    if not episode:
                        continue
                        
                    # Sample frames evenly
                    frame_indices = np.linspace(0, len(episode)-1, FRAMES_PER_EPISODE, dtype=int)
                    
                    for idx in frame_indices:
                        # Process single frame
                        frame = load_and_process_frame(episode[idx])
                        if frame is None:
                            continue
                            
                        # Extract features
                        features = extract_features(vit_model, vit_processor, frame, vit_feature_reducer)
                        if features is not None:
                            all_features.append(features)
                            game_labels.append(game_name)
                        
                        # Clear memory
                        clear_gpu_memory()
                    
                # Clean up episode data
                del episodes_data
                gc.collect()
                
                # Print memory status every 5 games
                if len(game_labels) % (5 * FRAMES_PER_EPISODE) == 0:
                    print(f"\nProcessed {len(game_labels)} frames")
                    print_memory_stats()
                
            except Exception as e:
                print(f"Error processing {game_name}: {e}")
                continue

        # Generate visualization
        if all_features:
            print("\nProcessing complete, generating visualization...")
            features_array = np.concatenate(all_features, axis=0)
            print(f"Number of features: {len(features_array)}")
            
            # Run t-SNE on CPU
            perplexity = min(30, len(features_array) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            features_2d = tsne.fit_transform(features_array)
            
            plt.figure(figsize=(15, 10))
            unique_games = list(set(game_labels))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_games)))
            
            for game, color in zip(unique_games, colors):
                mask = np.array(game_labels) == game
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          label=game.replace('Atari-', ''),
                          color=color, alpha=0.6, s=50)
            
            plt.title('Atari Games Feature Visualization (t-SNE)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.tight_layout()
            plt.savefig('atari_features_tsne_light.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Visualization saved as 'atari_features_tsne_light.png'")
            
        else:
            print("No features were successfully extracted")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Final cleanup
        clear_gpu_memory()
        print("Final memory state:")
        print_memory_stats()

if __name__ == "__main__":
    main()