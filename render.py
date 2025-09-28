import os

import imageio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_loader import NeRFDataLoader
from models import NeRFModel
from PIL import Image
from skimage import exposure
from tqdm import tqdm


def render(data_test, model, num_files):
    model.eval()
    frames = []
    with torch.no_grad():
        pred_colors_list = []

        for i in range(num_files):
            data = data_test.get_test_data(i)
            all_ray_directions = data["ray_directions"]
            all_points_on_rays = data["points_on_rays"]
            all_deltas = data["deltas"]

            # Process the image in chunks to avoid OOM errors
            chunk_size = 4096
            for i in tqdm(
                range(0, all_points_on_rays.shape[0], chunk_size), desc="Rendering"
            ):
                # Get chunk and move to GPU
                ray_dirs_chunk = all_ray_directions[i : i + chunk_size].cuda()
                points_chunk = all_points_on_rays[i : i + chunk_size].cuda()
                deltas_chunk = all_deltas[i : i + chunk_size].cuda()

                # Run model
                pred_chunk, _ = model(points_chunk, ray_dirs_chunk, deltas_chunk)
                pred_colors_list.append(pred_chunk.cpu())

        # Concatenate all chunks and reshape to an image
        pred_image_tensor = torch.cat(pred_colors_list, dim=0)

        pred_images_batch = pred_image_tensor.reshape(
            data_test.num_test_images, data_test.H, data_test.W, 3
        )
        for pred_image_tensor in pred_images_batch:
            frame = (pred_image_tensor.numpy() * 255).astype(np.uint8)
            frames.append(frame)

    # Save the frames as a video file
    video_path = f"video_renders/lego_video_pos_enc_best.mp4"
    imageio.mimsave(video_path, frames, fps=30, quality=8)
    print(f"Video saved to {video_path}")


def render_depth_map(data_test, model):
    model.eval()
    with torch.no_grad():

        for image_index in range(data_test.num_test_images):
            weights_chunks = []
            data = data_test.get_test_data(image_index)
            all_ray_directions = data["ray_directions"]
            all_points_on_rays = data["points_on_rays"]
            all_deltas = data["deltas"]
            dists_on_ray = data["dists_on_rays"]

            chunk_size = 4096
            for i in tqdm(
                range(0, all_points_on_rays.shape[0], chunk_size),
                desc="computing depth",
            ):
                ray_dirs_chunk = all_ray_directions[i : i + chunk_size].cuda()
                points_chunk = all_points_on_rays[i : i + chunk_size].cuda()
                deltas_chunk = all_deltas[i : i + chunk_size].cuda()

                _, weights = model(points_chunk, ray_dirs_chunk, deltas_chunk)

                weights_chunks.append(weights.cpu())

            weights_tensor = torch.cat(weights_chunks, dim=0)
            weights_image = weights_tensor.reshape(data_test.H, data_test.W, -1)
            dists_image = dists_on_ray.reshape(data_test.H, data_test.W, -1)

            depth_map = torch.sum(weights_image * dists_image, dim=-1)

            normalized_depth = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min()
            )

            # normalized_depth[current_mask] = 0.0
            img_np = (normalized_depth.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np, mode="L")
            img.save(f"renders/depth_map/frame_{image_index}.png")

    video_path = f"video_renders/lego_video_pos_enc_depth_map.mp4"
    with imageio.get_writer(video_path, fps=30) as writer:
        for i in range(data_test.num_test_images):
            writer.append_data(imageio.imread(f"renders/depth_map/frame_{i}.png"))
    print(f"Video saved to {video_path}")


def mask_and_normalize_depth_map(
    mask_video_path: str,
    content_video_path: str,
    output_video_path: str,
    white_threshold: int = 250,
):
    try:
        mask_reader = imageio.get_reader(mask_video_path)
        content_reader = imageio.get_reader(content_video_path)
    except FileNotFoundError:
        print("Error: One or both input video files not found.")
        return

    processed_frames = []

    for i in tqdm(range(100), desc="Processing frames"):
        mask_frame = mask_reader.get_data(i)
        content_frame = content_reader.get_data(i)

        white_mask = np.mean(mask_frame, axis=2) > white_threshold

        processed_frame = content_frame.copy()
        processed_frame[white_mask] = 0

        gamma = 1.5

        non_black_mask = processed_frame > 0
        foreground_pixels = processed_frame[non_black_mask]

        if foreground_pixels.size > 0:
            normalized_foreground = foreground_pixels / 255.0

            gamma_corrected_pixels = (
                1 / np.pi * np.arctan(10 * (normalized_foreground**gamma - 0.5)) + 0.5
            )

            final_pixels = (gamma_corrected_pixels * 255).astype(np.uint8)

            processed_frame[non_black_mask] = final_pixels

            processed_frame = (
                cm.viridis(processed_frame[:, :, 0] / 255.0)[:, :, :3] * 255
            )

            processed_frames.append(processed_frame.astype(np.uint8))

    # 4. Write the resulting frames to a new video file
    imageio.mimsave(output_video_path, processed_frames, fps=30, quality=8)
    print(f"Video saved to {output_video_path}")


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeRFModel(use_pos_enc=True).to(device)
    model.load_state_dict(
        torch.load(
            "checkpoints/nerf_experiment_pos_enc_1_coarse_model_pos_enc_step_70000.pt",
            map_location=device,
        )
    )

    data_test = NeRFDataLoader().init_test(num_test_images=100)
    render_depth_map(data_test, model)
    # render(data_test, model, 100)


main()
