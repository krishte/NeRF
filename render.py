import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_loader import NeRFDataLoader
from models import NeRFModel
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
                pred_chunk = model(points_chunk, ray_dirs_chunk, deltas_chunk)
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
    video_path = f"renders/lego_video.mp4"
    imageio.mimsave(video_path, frames, fps=30, quality=8)
    print(f"Video saved to {video_path}")


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeRFModel().to(device)
    model.load_state_dict(
        torch.load(
            "checkpoints/nerf_experiment_9_model_step_100000.pt", map_location=device
        )
    )

    data_test = NeRFDataLoader().init_test(num_test_images=100)
    render(data_test, model, 100)


main()
