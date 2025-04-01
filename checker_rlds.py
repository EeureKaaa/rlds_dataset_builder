import tensorflow_datasets as tfds
import tensorflow as tf
import os
import pprint
import numpy as np

# Original dataset names list
RAW_DATASET_NAMES = [
    "close_door_dataset",
]  
data_dir = "/home/wangxianhao/data/project/reasoning/rlds_dataset_builder/tabletop_dataset"

# Dictionary to store dataset names, episodes, and step counts
num_dict = {}

def print_tensor_info(name, tensor):
    """Print information about a tensor including shape and dtype."""
    if isinstance(tensor, np.ndarray):
        print(f"  {name} shape: {tensor.shape}, dtype: {tensor.dtype}")
    elif isinstance(tensor, (str, bytes)):
        print(f"  {name}: {type(tensor)} (length: {len(tensor)})")
    elif isinstance(tensor, (int, float, bool)):
        print(f"  {name}: {type(tensor)} (value: {tensor})")
    else:
        print(f"  {name}: {type(tensor)}")

def analyze_rlds_dataset(dataset_path):
    """Analyze a specific RLDS dataset file."""
    print(f"\nAnalyzing RLDS dataset at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"File not found: {dataset_path}")
        return
    
    # Load the dataset using TensorFlow's TFRecordDataset
    raw_dataset = tf.data.TFRecordDataset([dataset_path])
    
    # Create a feature description for parsing RLDS format
    feature_description = {
        'episode_metadata': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'steps': tf.io.VarLenFeature(tf.string),
    }
    
    def _parse_episode(example_proto):
        # Parse the input tf.Example proto
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        return parsed_features
    
    # Parse the dataset
    parsed_dataset = raw_dataset.map(_parse_episode)
    
    # Analyze the dataset
    episode_count = 0
    for episode_idx, episode in enumerate(parsed_dataset):
        episode_count += 1
        
        # Only analyze the first episode in detail
        if episode_idx == 0:
            print("\n--- First Episode Analysis ---")
            
            # Parse episode metadata
            if 'episode_metadata' in episode:
                try:
                    metadata = tf.io.parse_tensor(episode['episode_metadata'], tf.string)
                    print(f"Episode metadata: {metadata.numpy()}")
                except:
                    print(f"Episode metadata: {type(episode['episode_metadata'])}")
            
            # Parse and analyze steps
            if 'steps' in episode:
                steps_sparse = episode['steps']
                steps_dense = tf.sparse.to_dense(steps_sparse)
                
                # Count steps
                num_steps = steps_dense.shape[0]
                print(f"Number of steps in first episode: {num_steps}")
                
                # Analyze first step in detail
                if num_steps > 0:
                    try:
                        # Parse the first step
                        first_step = tf.io.parse_tensor(steps_dense[0], tf.string)
                        step_dict = tf.io.parse_tensor(first_step, tf.string)
                        print("\n--- First Step Analysis ---")
                        print(f"Step keys: {step_dict.keys()}")
                        
                        # Analyze step components
                        for key in step_dict.keys():
                            print(f"\n{key}:")
                            if key == 'observation' and isinstance(step_dict[key], dict):
                                for obs_key, obs_value in step_dict[key].items():
                                    print_tensor_info(obs_key, obs_value)
                            else:
                                print_tensor_info(key, step_dict[key])
                    except Exception as e:
                        print(f"Error parsing step: {e}")
                        # Try a different approach
                        print("Attempting alternative parsing method...")
                        try:
                            step_example = tf.train.Example.FromString(steps_dense[0].numpy())
                            print(f"Step example features: {step_example.features.feature.keys()}")
                        except Exception as e2:
                            print(f"Alternative parsing also failed: {e2}")
        
        # Stop after analyzing a few episodes
        if episode_idx >= 2:
            print(f"\nAnalyzed {episode_idx + 1} episodes, stopping for brevity.")
            break
    
    print(f"\nTotal episodes found: {episode_count}")

# Check the specific dataset file
specific_file_path = "/home/wangxianhao/data/project/reasoning/rlds_dataset_builder/tabletop_dataset/close_door_dataset/1.0.0/close_door_dataset-train.tfrecord-00000-of-00004"
analyze_rlds_dataset(specific_file_path)

# Try loading using tfds.load for comparison
print("\n--- Attempting to load with tfds.load ---")
try:
    # Try to load the dataset using the standard TFDS approach
    dataset_name = RAW_DATASET_NAMES[0]
    dataset = tfds.load(
        dataset_name,
        data_dir=data_dir,
        split="train",
        shuffle_files=False,
    )
    
    print(f"Successfully loaded {dataset_name} with tfds.load")
    
    # Analyze the first few episodes
    episode_count = 0
    for episode_idx, episode in enumerate(dataset):
        episode_count += 1
        
        if episode_idx == 0:
            print("\n--- First Episode Structure (tfds.load) ---")
            print(f"Episode keys: {list(episode.keys())}")
            
            if "steps" in episode:
                # Get the first step
                for step_idx, step in enumerate(episode["steps"]):
                    print(f"\n--- Step {step_idx} ---")
                    print(f"Step keys: {list(step.keys())}")
                    
                    if "observation" in step:
                        print(f"Observation keys: {list(step['observation'].keys())}")
                        for obs_key, obs_value in step["observation"].items():
                            if hasattr(obs_value, "shape"):
                                print(f"  {obs_key} shape: {obs_value.shape}, dtype: {obs_value.dtype}")
                            else:
                                print(f"  {obs_key}: {type(obs_value)}")
                    
                    if "action" in step:
                        if hasattr(step["action"], "shape"):
                            print(f"Action shape: {step['action'].shape}, dtype: {step['action'].dtype}")
                        else:
                            print(f"Action: {type(step['action'])}")
                    
                    # Only show first step
                    break
        
        # Only show a few episodes
        if episode_idx >= 2:
            break
    
    print(f"\nTotal episodes found with tfds.load: {episode_count}")
    
except Exception as e:
    print(f"Error loading with tfds.load: {e}")

print("\nAnalysis complete.")