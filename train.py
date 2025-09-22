import itertools

import numpy as np
import torch
from data_loader import NeRFDataLoader
from models import NeRFModel
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def psnr(pred, target):
    """Calculates Peak Signal-to-Noise Ratio."""
    mse = torch.mean((pred - target) ** 2)
    return 10.0 * torch.log10(1.0 / mse)


def train_coarse_and_fine(
    data_train: NeRFDataLoader,
    coarse_model: NeRFModel,
    fine_model: NeRFModel,
    criterion,
    optimizer,
    scheduler,
    global_step,
    writer,
    iterations=10000,
):
    coarse_model.train()
    fine_model.train()
    for i in tqdm(range(iterations)):
        data = data_train.get_data()
        (ray_directions, points_on_rays, colors, deltas, ray_origins, dists_on_rays) = (
            data["ray_directions"].cuda(),
            data["points_on_rays"].cuda(),
            data["real_colors"].cuda(),
            data["deltas"].cuda(),
            data["ray_origins"].cuda(),
            data["dists_on_rays"].cuda(),
        )
        _, weights = coarse_model(points_on_rays, ray_directions, deltas)

        weights_detached = weights.detach()

        (deltas_fine, points_on_rays_fine) = data_train.get_fine_data(
            ray_origins, ray_directions, dists_on_rays, weights_detached
        )

        pred_colors_fine, _ = fine_model(
            points_on_rays_fine, ray_directions, deltas_fine
        )
        fine_loss = criterion(pred_colors_fine, colors)
        optimizer.zero_grad()

        writer.add_scalar("Loss/fine_train", fine_loss.item(), global_step)

        fine_loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1
    return global_step


def train(
    data_train: NeRFDataLoader,
    model: NeRFModel,
    criterion,
    optimizer,
    scheduler,
    global_step,
    writer,
    iterations=10000,
):
    model.train()
    for i in tqdm(range(iterations)):
        data = data_train.get_data()
        (
            ray_directions,
            points_on_rays,
            colors,
            deltas,
        ) = (
            data["ray_directions"].cuda(),
            data["points_on_rays"].cuda(),
            data["real_colors"].cuda(),
            data["deltas"].cuda(),
        )
        optimizer.zero_grad()
        pred_colors = model(points_on_rays, ray_directions, deltas)

        loss = criterion(pred_colors, colors)
        writer.add_scalar("Loss/train", loss.item(), global_step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # Add this line
        optimizer.step()
        scheduler.step()
        global_step += 1
    return global_step


def validate(data_val, model, criterion, step, writer):
    model.eval()
    with torch.no_grad():
        # Get all rays for the single validation image
        data = data_val.get_data()
        all_ray_directions = data["ray_directions"]
        all_points_on_rays = data["points_on_rays"]
        all_real_colors = data["real_colors"]
        all_deltas = data["deltas"]

        pred_colors_list = []

        # Process the image in chunks to avoid OOM errors
        chunk_size = 4096
        for i in tqdm(
            range(0, all_points_on_rays.shape[0], chunk_size), desc="Validating"
        ):
            # Get chunk and move to GPU
            ray_dirs_chunk = all_ray_directions[i : i + chunk_size].cuda()
            points_chunk = all_points_on_rays[i : i + chunk_size].cuda()
            deltas_chunk = all_deltas[i : i + chunk_size].cuda()

            # Run model
            pred_chunk = model(points_chunk, ray_dirs_chunk, deltas_chunk)

            # Append results (move back to CPU to save GPU memory)
            pred_colors_list.append(pred_chunk.cpu())

        # Concatenate all chunks and reshape to an image
        pred_image_tensor = torch.cat(pred_colors_list, dim=0)

        pred_images_batch = pred_image_tensor.reshape(
            data_val.num_val_images, data_val.H, data_val.W, 3
        )
        real_images_batch = all_real_colors.reshape(
            data_val.num_val_images, data_val.H, data_val.W, 3
        )

        # Calculate metrics
        val_loss = criterion(pred_images_batch, real_images_batch)
        val_psnr = psnr(pred_images_batch, real_images_batch)

        if writer:
            writer.add_scalar("Loss/validation", val_loss.item(), step)
            writer.add_scalar("PSNR/validation", val_psnr.item(), step)

        print(f"Validation Loss: {val_loss.item():.4f}, PSNR: {val_psnr.item():.2f}")

        for i in range(data_val.num_val_images):
            pred_image_tensor = pred_images_batch[i]

            # Convert to a savable format (numpy array)
            pred_image_np = (pred_image_tensor.numpy() * 255).astype(np.uint8)

            # Save the rendered image to disk with a unique name
            img = Image.fromarray(pred_image_np)
            img.save(f"renders/step_{step:06d}_img_{i:02d}.png")

            # Add each image to TensorBoard under a unique tag
            if writer:
                writer.add_image(
                    f"Validation Renders/Image_{i}",
                    pred_image_tensor.permute(2, 0, 1),
                    step,
                )


def main():
    name = "nerf_experiment_fine_2"
    writer = SummaryWriter(f"runs/{name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_train = NeRFDataLoader().init_train("lego")
    data_val = NeRFDataLoader().init_val("lego")

    coarse_model = NeRFModel().to(device)
    fine_model = NeRFModel().to(device)
    coarse_model.load_state_dict(
        torch.load(
            "checkpoints/nerf_experiment_9_model_step_10000.pt", map_location=device
        )
    )
    for param in coarse_model.parameters():
        param.requires_grad = False

    fine_model.load_state_dict(
        torch.load(
            "checkpoints/nerf_experiment_9_model_step_10000.pt", map_location=device
        )
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(fine_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99998848714)

    global_step = 0
    for i in range(20):
        global_step = train_coarse_and_fine(
            data_train,
            coarse_model,
            fine_model,
            criterion,
            optimizer,
            scheduler,
            global_step,
            writer,
        )
        coarse_checkpoint_path = (
            f"checkpoints/{name}_coarse_model_step_{global_step}.pt"
        )
        fine_checkpoint_path = f"checkpoints/{name}_fine_model_step_{global_step}.pt"
        torch.save(coarse_model.state_dict(), coarse_checkpoint_path)
        torch.save(fine_model.state_dict(), fine_checkpoint_path)

        validate(data_val, fine_model, criterion, global_step, writer)


if __name__ == "__main__":
    main()
