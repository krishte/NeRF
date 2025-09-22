import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


class NeRFDataset:
    def __init__(self, dataset_dir, split):
        self.dataset_dir = dataset_dir
        self.split = split

    def load_data_for_object(self, object_name):
        camera_info = []
        file_paths = []
        camera_angle_x = None
        json_path = f"{self.dataset_dir}/{object_name}/transforms_{self.split}.json"
        with open(json_path, "r") as f:
            data = json.load(f)
            camera_angle_x = data["camera_angle_x"]
            for frame in data["frames"]:
                camera_info.append(frame["transform_matrix"])
                file_paths.append(
                    os.path.join(
                        self.dataset_dir,
                        object_name,
                        frame["file_path"].lstrip("./\\") + ".png",
                    )
                )
        print(f"Number of files for {self.split}", len(file_paths))
        return camera_info, file_paths, camera_angle_x


class NeRFDataLoader:
    def __init__(self, num_coarse_samples=128, num_fine_samples=64):
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples

    def init_train(self, object, batch_size=4096, load_chunk_size=500):
        self.split = "train"
        self.H = self.W = 800
        self.batch_size = batch_size
        self.load_chunk_size = load_chunk_size

        dataset = NeRFDataset(
            dataset_dir="datasets/nerf_synthetic/nerf_synthetic", split="train"
        )
        self.camera_info, file_paths, camera_angle_x = dataset.load_data_for_object(
            object
        )
        self.store_pixel_direction_vectors(camera_angle_x)
        self.store_image_pixels(file_paths)

        self.current_batch_index = -1

        return self

    def init_val(self, object, num_val_images=5):
        self.split = "val"
        self.H = self.W = 800
        self.num_val_images = num_val_images
        self.batch_size = self.num_val_images * self.H * self.W
        self.load_chunk_size = 1

        dataset = NeRFDataset(
            dataset_dir="datasets/nerf_synthetic/nerf_synthetic", split="val"
        )
        self.camera_info, file_paths, camera_angle_x = dataset.load_data_for_object(
            object
        )
        self.store_pixel_direction_vectors(camera_angle_x)
        self.store_image_pixels(file_paths)

        return self

    def init_test(
        self,
        num_test_images=100,
        theta=np.pi * 5 / 12,
        video_radius=4.0,
        H=800,
        W=800,
        camera_angle_x=0.6911112070083618,
    ):
        self.split = "test"
        self.H, self.W = H, W
        self.num_test_images = num_test_images
        self.batch_size = self.H * self.W
        self.load_chunk_size = 1

        self.camera_info = self.generate_test_camera_data(theta, video_radius)
        self.store_pixel_direction_vectors(camera_angle_x)

        return self

    def generate_test_camera_data(self, theta, video_radius):
        theta = np.repeat(theta, self.num_test_images)
        phi = np.linspace(0.0, 2 * np.pi, self.num_test_images)

        camera_pos_x = video_radius * np.sin(theta) * np.cos(phi)
        camera_pos_y = video_radius * np.sin(theta) * np.sin(phi)
        camera_pos_z = video_radius * np.cos(theta)

        camera_pos = np.stack([camera_pos_x, camera_pos_y, camera_pos_z], axis=1)

        camera_z_axis = camera_pos / np.linalg.norm(camera_pos, axis=-1, keepdims=True)

        camera_up = np.tile(np.array([0.0, 0.0, 1.0]), (self.num_test_images, 1))

        camera_x_axis = np.cross(camera_up, camera_z_axis)
        camera_x_axis = camera_x_axis / np.linalg.norm(
            camera_x_axis, axis=-1, keepdims=True
        )

        camera_y_axis = np.cross(camera_z_axis, camera_x_axis)
        camera_y_axis = camera_y_axis / np.linalg.norm(
            camera_y_axis, axis=-1, keepdims=True
        )

        camera_matrix = np.stack(
            [camera_x_axis, camera_y_axis, camera_z_axis, camera_pos], axis=1
        )
        camera_matrix = np.transpose(camera_matrix, (0, 2, 1))

        return camera_matrix

    def store_image_pixels(self, file_paths):
        self.image_pixels = np.array(
            [
                np.array(Image.open(file_path).convert("RGBA")) / 255.0
                for file_path in file_paths
            ]
        )

    def store_pixel_direction_vectors(self, camera_angle_x):
        f_x = f_y = self.W / (2 * np.tan(camera_angle_x / 2))
        c_x = self.W / 2
        c_y = self.H / 2
        self.pixel_direction_vectors = self.build_pixel_direction_vectors(
            self.H, self.W, f_x, f_y, c_x, c_y
        )

    def build_pixel_direction_vectors(self, H, W, f_x, f_y, c_x, c_y):
        u_array = np.arange(W) + 0.5
        v_array = np.arange(H) + 0.5
        u_grid, v_grid = np.meshgrid(u_array, v_array, indexing="xy")

        x_cam = (u_grid - c_x) / f_x
        y_cam = -(v_grid - c_y) / f_y
        d_cam = np.stack([x_cam, y_cam, -np.ones_like(x_cam)], axis=-1)
        d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
        return d_cam

    def sample_points_from_rays(self, file_indices, pixel_selections):

        (
            self.colors,
            self.points_on_rays,
            self.dists_on_rays,
            self.ray_directions,
            self.deltas,
        ) = (
            np.empty((self.batch_size * self.load_chunk_size, 3), dtype=np.float32),
            np.empty(
                (self.batch_size * self.load_chunk_size, self.num_coarse_samples, 3),
                dtype=np.float32,
            ),
            np.empty(
                (self.batch_size * self.load_chunk_size, self.num_coarse_samples),
                dtype=np.float32,
            ),
            np.empty((self.batch_size * self.load_chunk_size, 3), dtype=np.float32),
            np.empty(
                (self.batch_size * self.load_chunk_size, self.num_coarse_samples),
                dtype=np.float32,
            ),
        )
        pixel_v_selections = pixel_selections // self.W
        pixel_u_selections = pixel_selections % self.W

        camera_infos = np.array(self.camera_info)[file_indices]

        ray_origins = camera_infos[:, :3, 3]
        self.ray_origins = ray_origins
        ray_directions = np.einsum(
            "nij,nj->ni",
            camera_infos[:, :3, :3],
            self.pixel_direction_vectors[pixel_v_selections, pixel_u_selections],
        )
        self.ray_directions = ray_directions / np.linalg.norm(
            ray_directions, axis=1, keepdims=True
        )

        t_close, t_far = 2.0, 6.0
        num_rays = self.batch_size * self.load_chunk_size
        evenly_spaced_dists = np.tile(
            np.linspace(t_close, t_far, self.num_coarse_samples + 1), (num_rays, 1)
        )
        self.dists_on_rays = evenly_spaced_dists[:, :-1] + (
            evenly_spaced_dists[:, 1:] - evenly_spaced_dists[:, :-1]
        ) * np.random.random((num_rays, self.num_coarse_samples))

        self.points_on_rays = (
            ray_origins[:, np.newaxis, :]
            + self.dists_on_rays[..., np.newaxis]
            * self.ray_directions[:, np.newaxis, :]
        )

        deltas = self.dists_on_rays[:, 1:] - self.dists_on_rays[:, :-1]
        last_delta = 1e10 * np.ones((num_rays, 1))
        self.deltas = np.concatenate([deltas, last_delta], axis=1)

        if self.split == "train" or self.split == "val":
            selected_colors = self.image_pixels[
                file_indices, pixel_v_selections, pixel_u_selections
            ]

            rgb = selected_colors[:, :3]
            a = selected_colors[:, 3:4]
            self.colors = rgb * a + (1.0 - a)  # composite over white background

    def sample_files(self, index=None):
        file_indices, pixel_selections = None, None
        num_pixels = self.W * self.H

        if self.split == "train":
            file_indices = np.random.randint(
                0, len(self.camera_info), size=self.batch_size * self.load_chunk_size
            )
            pixel_selections = np.random.randint(
                0, self.W * self.H, size=self.batch_size * self.load_chunk_size
            )
        elif self.split == "val":
            image_indices = np.arange(self.num_val_images)
            file_indices = np.repeat(image_indices, num_pixels)
            single_image_pixels = np.arange(num_pixels)
            pixel_selections = np.tile(single_image_pixels, self.num_val_images)

        elif self.split == "test":
            image_indices = np.array([index])
            file_indices = np.repeat(image_indices, num_pixels)
            pixel_selections = np.arange(num_pixels)

        return file_indices, pixel_selections

    def get_test_data(self, index):
        file_indices, pixel_selections = self.sample_files(index)
        self.sample_points_from_rays(file_indices, pixel_selections)
        return {
            "ray_directions": torch.from_numpy(self.ray_directions).float(),
            "points_on_rays": torch.from_numpy(self.points_on_rays).float(),
            "dists_on_rays": torch.from_numpy(self.dists_on_rays).float(),
            "deltas": torch.from_numpy(self.deltas).float(),
        }

    def get_fine_data(self, ray_origins, ray_directions, dists_on_rays_coarse, weights):
        weights = weights + 1e-5
        pdf = weights / torch.sum(weights, dim=-1, keepdims=True)
        cdf = torch.cumsum(pdf, dim=-1)
        # Prepend a zero to the CDF for the first bin
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        randoms = torch.rand(
            (self.batch_size, self.num_fine_samples), device=cdf.device
        )

        indices = torch.searchsorted(cdf, randoms, right=True)
        indices = torch.clamp(indices, max=cdf.shape[-1] - 1)

        # Gather the lower and upper bounds for each bin where a random sample fell
        max_valid_index = dists_on_rays_coarse.shape[-1] - 1

        below = torch.max(torch.zeros_like(indices - 1), indices - 1)
        above = torch.min(torch.full_like(indices, max_valid_index), indices)

        cdf_below = torch.gather(cdf, dim=-1, index=below)
        cdf_above = torch.gather(cdf, dim=-1, index=above)
        dists_coarse_below = torch.gather(dists_on_rays_coarse, dim=-1, index=below)
        dists_coarse_above = torch.gather(dists_on_rays_coarse, dim=-1, index=above)

        # Perform linear interpolation to get the fine sample distances
        denom = cdf_above - cdf_below
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (randoms - cdf_below) / denom

        dists_on_rays_fine = dists_coarse_below + t * (
            dists_coarse_above - dists_coarse_below
        )

        dists_on_rays_all, _ = torch.sort(
            torch.cat([dists_on_rays_coarse, dists_on_rays_fine], dim=-1), dim=-1
        )
        deltas = dists_on_rays_all[:, 1:] - dists_on_rays_all[:, :-1]
        last_delta = torch.full((deltas.shape[0], 1), 1e10, device=deltas.device)
        deltas = torch.cat([deltas, last_delta], dim=-1)

        points_on_rays = (
            ray_origins[..., None, :]
            + ray_directions[..., None, :] * dists_on_rays_all[..., :, None]
        )

        return deltas, points_on_rays

    def get_data(self):
        if self.split == "val" or self.split == "test":
            file_indices, pixel_selections = self.sample_files()
            self.sample_points_from_rays(file_indices, pixel_selections)
            return_val = {
                "ray_directions": torch.from_numpy(self.ray_directions).float(),
                "points_on_rays": torch.from_numpy(self.points_on_rays).float(),
                "dists_on_rays": torch.from_numpy(self.dists_on_rays).float(),
                "deltas": torch.from_numpy(self.deltas).float(),
            }
            if self.split == "val":
                return_val["real_colors"] = torch.from_numpy(self.colors).float()
            return return_val

        self.current_batch_index += 1
        if (
            self.split == "train"
            and self.current_batch_index % self.load_chunk_size == 0
        ):
            self.current_batch_index = 0
            file_indices, pixel_selections = self.sample_files()
            self.sample_points_from_rays(file_indices, pixel_selections)

        return {
            "ray_directions": torch.from_numpy(
                self.ray_directions[
                    self.current_batch_index
                    * self.batch_size : (self.current_batch_index + 1)
                    * self.batch_size
                ]
            ).float(),  # Shape (batch_size, 3)
            "points_on_rays": torch.from_numpy(
                self.points_on_rays[
                    self.current_batch_index
                    * self.batch_size : (self.current_batch_index + 1)
                    * self.batch_size
                ]
            ).float(),  # Shape (batch_size, num_samples, 3)
            "dists_on_rays": torch.from_numpy(
                self.dists_on_rays[
                    self.current_batch_index
                    * self.batch_size : (self.current_batch_index + 1)
                    * self.batch_size
                ]
            ).float(),  # Shape (batch_size, num_samples)
            "real_colors": torch.from_numpy(
                self.colors[
                    self.current_batch_index
                    * self.batch_size : (self.current_batch_index + 1)
                    * self.batch_size
                ]
            ).float(),  # Shape (batch_size, 3)
            "ray_origins": torch.from_numpy(
                self.ray_origins[
                    self.current_batch_index
                    * self.batch_size : (self.current_batch_index + 1)
                    * self.batch_size
                ]
            ).float(),
            "deltas": torch.from_numpy(
                self.deltas[
                    self.current_batch_index
                    * self.batch_size : (self.current_batch_index + 1)
                    * self.batch_size
                ]
            ).float(),  # Shape (batch_size, num_samples)
        }
