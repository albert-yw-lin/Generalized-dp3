import zarr
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import argparse
from tqdm import tqdm

def load_zarr_dataset(zarr_path):
    """Load zarr dataset and return data arrays"""
    root = zarr.open(zarr_path, mode='r')
    
    data = {}
    data['img'] = root['data']['img']
    data['depth'] = root['data']['depth'] 
    data['point_cloud'] = root['data']['point_cloud']
    data['action'] = root['data']['action']
    data['state'] = root['data']['state']
    
    print(f"Dataset loaded from: {zarr_path}")
    print(f"Image shape: {data['img'].shape}")
    print(f"Depth shape: {data['depth'].shape}")
    print(f"Point cloud shape: {data['point_cloud'].shape}")
    print(f"Action shape: {data['action'].shape}")
    print(f"State shape: {data['state'].shape}")
    
    # Check if this is a flat dataset (each row is a timestep) or episodic
    if len(data['img'].shape) == 4:  # (N, H, W, C) - flat format
        print("Detected flat dataset format (each row is a timestep)")
        data['format'] = 'flat'
    else:  # (N_episodes, episode_length, H, W, C) - episodic format
        print("Detected episodic dataset format")
        data['format'] = 'episodic'
    
    return data

def create_visualization_frame(img, depth, action, state, episode_idx, step_idx):
    """Create a single frame combining image, depth, and info"""
    # Convert image from RGB to BGR for OpenCV (fix color channel order)
    if img.max() <= 1.0:
        img_vis = (img * 255).astype(np.uint8)
    else:
        img_vis = img.astype(np.uint8)
    
    # Convert RGB to BGR for correct colors in OpenCV
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    
    # Normalize depth for visualization
    if depth.max() > 1.0:
        depth_norm = (depth / depth.max() * 255).astype(np.uint8)
    else:
        depth_norm = (depth * 255).astype(np.uint8)
    
    # Convert depth to 3-channel for concatenation
    depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    # Resize images to same height if needed
    h1, w1 = img_vis.shape[:2]
    h2, w2 = depth_vis.shape[:2]
    
    if h1 != h2:
        target_height = min(h1, h2)
        img_vis = cv2.resize(img_vis, (int(w1 * target_height / h1), target_height))
        depth_vis = cv2.resize(depth_vis, (int(w2 * target_height / h2), target_height))
    
    # Concatenate images horizontally
    combined = np.hstack([img_vis, depth_vis])
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White text
    thickness = 2
    
    # Add episode and step info
    cv2.putText(combined, f"Episode: {episode_idx}, Step: {step_idx}", 
                (10, 30), font, font_scale, color, thickness)
    
    # Add action information
    action_text = f"Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}]"
    cv2.putText(combined, action_text, 
                (10, 60), font, font_scale * 0.8, color, thickness)
    
    # Add gripper state if available
    if len(action) > 4:
        gripper_text = f"Gripper: {action[4]:.3f}"
        cv2.putText(combined, gripper_text, 
                    (10, 90), font, font_scale * 0.8, color, thickness)
    
    return combined

def visualize_episode(data, episode_idx, output_dir, fps=10, episode_length=200):
    """Visualize a single episode and save as video"""
    if data['format'] == 'flat':
        # For flat format, we need to extract a sequence of frames
        start_idx = episode_idx * episode_length
        end_idx = min(start_idx + episode_length, data['img'].shape[0])
        
        episode_imgs = data['img'][start_idx:end_idx]
        episode_depths = data['depth'][start_idx:end_idx]
        episode_actions = data['action'][start_idx:end_idx]
        episode_states = data['state'][start_idx:end_idx]
        actual_length = end_idx - start_idx
    else:
        # Original episodic format
        actual_length = data['img'].shape[1]
        episode_imgs = data['img'][episode_idx]
        episode_depths = data['depth'][episode_idx]
        episode_actions = data['action'][episode_idx]
        episode_states = data['state'][episode_idx]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup video writer
    sample_frame = create_visualization_frame(
        episode_imgs[0], episode_depths[0], 
        episode_actions[0], episode_states[0], 
        episode_idx, 0
    )
    
    height, width = sample_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_dir, f'episode_{episode_idx:03d}.mp4')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    print(f"Creating video for episode {episode_idx} (length: {actual_length})...")
    
    for step_idx in tqdm(range(actual_length), desc=f"Episode {episode_idx}"):
        frame = create_visualization_frame(
            episode_imgs[step_idx], 
            episode_depths[step_idx],
            episode_actions[step_idx],
            episode_states[step_idx],
            episode_idx, 
            step_idx
        )
        
        out.write(frame)
    
    out.release()
    print(f"Video saved: {video_path}")
    
    return video_path

def create_summary_video(data, output_dir, num_episodes=5, fps=10, episode_length=200):
    """Create a summary video showing multiple episodes"""
    if data['format'] == 'flat':
        total_episodes = min(num_episodes, data['img'].shape[0] // episode_length)
    else:
        total_episodes = data['img'].shape[0]
    
    episodes_to_show = min(num_episodes, total_episodes)
    
    print(f"Creating summary video with {episodes_to_show} episodes...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect frames from multiple episodes
    all_frames = []
    
    for ep_idx in range(episodes_to_show):
        if data['format'] == 'flat':
            start_idx = ep_idx * episode_length
            end_idx = min(start_idx + episode_length, data['img'].shape[0])
            
            episode_imgs = data['img'][start_idx:end_idx]
            episode_depths = data['depth'][start_idx:end_idx]
            episode_actions = data['action'][start_idx:end_idx]
            episode_states = data['state'][start_idx:end_idx]
            actual_length = end_idx - start_idx
        else:
            actual_length = data['img'].shape[1]
            episode_imgs = data['img'][ep_idx]
            episode_depths = data['depth'][ep_idx]
            episode_actions = data['action'][ep_idx]
            episode_states = data['state'][ep_idx]
        
        # Sample frames from episode (every 5th frame to keep video manageable)
        sample_indices = range(0, actual_length, 5)
        
        for step_idx in sample_indices:
            frame = create_visualization_frame(
                episode_imgs[step_idx], 
                episode_depths[step_idx],
                episode_actions[step_idx],
                episode_states[step_idx],
                ep_idx, 
                step_idx
            )
            all_frames.append(frame)
    
    # Write summary video
    if all_frames:
        height, width = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        summary_path = os.path.join(output_dir, 'summary.mp4')
        out = cv2.VideoWriter(summary_path, fourcc, fps, (width, height))
        
        for frame in tqdm(all_frames, desc="Writing summary video"):
            out.write(frame)
        
        out.release()
        print(f"Summary video saved: {summary_path}")
        return summary_path
    
    return None

def analyze_demonstration_quality(data, episode_length=200):
    """Analyze and print statistics about the demonstration quality"""
    print("\n" + "="*50)
    print("DEMONSTRATION QUALITY ANALYSIS")
    print("="*50)
    
    if data['format'] == 'flat':
        total_timesteps = data['img'].shape[0]
        num_episodes = total_timesteps // episode_length
        print(f"Total timesteps: {total_timesteps}")
        print(f"Estimated episodes (assuming length {episode_length}): {num_episodes}")
        print(f"Episode length: {episode_length}")
    else:
        num_episodes = data['img'].shape[0]
        episode_length = data['img'].shape[1]
        print(f"Number of episodes: {num_episodes}")
        print(f"Episode length: {episode_length}")
    
    # Analyze actions
    actions = np.array(data['action'])
    print(f"\nAction statistics:")
    print(f"  Shape: {actions.shape}")
    print(f"  Mean: {np.mean(actions, axis=0)}")
    print(f"  Std: {np.std(actions, axis=0)}")
    print(f"  Min: {np.min(actions, axis=0)}")
    print(f"  Max: {np.max(actions, axis=0)}")
    
    # Analyze action smoothness (consecutive differences)
    action_diffs = np.diff(actions, axis=0)
    print(f"\nAction smoothness (consecutive differences):")
    print(f"  Mean abs diff: {np.mean(np.abs(action_diffs), axis=0)}")
    print(f"  Max abs diff: {np.max(np.abs(action_diffs), axis=0)}")
    
    if data['format'] == 'flat':
        # For flat format, analyze episode-wise statistics
        episode_action_vars = []
        for ep_idx in range(num_episodes):
            start_idx = ep_idx * episode_length
            end_idx = min(start_idx + episode_length, actions.shape[0])
            if end_idx - start_idx > 1:
                ep_actions = actions[start_idx:end_idx]
                ep_var = np.var(ep_actions, axis=0)
                episode_action_vars.append(ep_var)
        
        if episode_action_vars:
            episode_action_vars = np.array(episode_action_vars)
            static_episodes = np.sum(np.all(episode_action_vars < 0.001, axis=1))
            print(f"\nPotentially static episodes: {static_episodes}/{num_episodes}")
    else:
        # Original episodic format
        action_vars = np.var(actions, axis=1)
        static_episodes = np.sum(np.all(action_vars < 0.001, axis=1))
        print(f"\nPotentially static episodes: {static_episodes}/{num_episodes}")
    
    # Analyze states
    states = np.array(data['state'])
    print(f"\nState statistics:")
    print(f"  Shape: {states.shape}")
    print(f"  Mean: {np.mean(states, axis=0)}")
    print(f"  Std: {np.std(states, axis=0)}")

def main():
    parser = argparse.ArgumentParser(description='Visualize MetaWorld demonstrations')
    parser.add_argument('--zarr_path', type=str, 
                       default='3D-Diffusion-Policy/data/metaworld_button-press-wall_expert.zarr',
                       help='Path to zarr dataset')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                       help='Output directory for videos')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to visualize')
    parser.add_argument('--episode_length', type=int, default=200,
                       help='Length of each episode (for flat format datasets)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Video frame rate')
    parser.add_argument('--summary_only', action='store_true',
                       help='Only create summary video, not individual episodes')
    
    args = parser.parse_args()
    
    # Load dataset
    data = load_zarr_dataset(args.zarr_path)
    
    # Analyze demonstration quality
    analyze_demonstration_quality(data, args.episode_length)
    
    # Create visualizations
    if args.summary_only:
        create_summary_video(data, args.output_dir, args.episodes, args.fps, args.episode_length)
    else:
        # Create individual episode videos
        if data['format'] == 'flat':
            total_episodes = min(args.episodes, data['img'].shape[0] // args.episode_length)
        else:
            total_episodes = min(args.episodes, data['img'].shape[0])
            
        for ep_idx in range(total_episodes):
            visualize_episode(data, ep_idx, args.output_dir, args.fps, args.episode_length)
        
        # Also create summary video
        create_summary_video(data, args.output_dir, args.episodes, args.fps, args.episode_length)
    
    print(f"\nVisualization complete! Check {args.output_dir} for output videos.")

if __name__ == "__main__":
    main() 