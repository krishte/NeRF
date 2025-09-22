import torch
import torch.nn.functional as F


class NeRFModel(torch.nn.Module):
    def __init__(self, hidden=256):
        super(NeRFModel, self).__init__()

        self.seq1 = torch.nn.Sequential(
            torch.nn.Linear(3, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
        )

        self.seq2 = torch.nn.Sequential(
            torch.nn.Linear(hidden + 3, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
        )

        self.density_head = torch.nn.Linear(hidden, 1)
        self.color_features = torch.nn.Linear(hidden, hidden)

        self.color_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden + 3, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 3),
            torch.nn.Sigmoid(),
        )

    def forward(
        self,
        points_on_ray,
        ray_direction,
        deltas,
        bg_color=(1.0, 1.0, 1.0),
    ):
        """
        points_on_ray: (batch_size, num_samples, 3) tensor
        ray_direction: (batch_size, 3) tensor
        valid_mask: (batch_size, num_samples) tensor
        deltas: (batch_size, num_samples) tensor
        """
        batch_size, num_samples, _ = points_on_ray.shape

        flattened_points = points_on_ray.reshape(-1, 3)
        repeated_ray_direction = (
            ray_direction[:, None, :]
            .expand(batch_size, num_samples, 3)  # Use dynamic shapes
            .reshape(-1, 3)
        )

        h = self.seq1(flattened_points)
        h = torch.cat([h, flattened_points], dim=-1)
        h = self.seq2(h)

        sigma = F.relu(self.density_head(h).squeeze(-1))
        sigmas = sigma.view(batch_size, num_samples)
        color_features = self.color_features(h)
        h = torch.cat([color_features, repeated_ray_direction], dim=-1)
        colors = self.color_classifier(h).view(batch_size, num_samples, 3)

        sigma_deltas = sigmas * deltas  # (batch_size, num_samples)
        alphas = 1.0 - torch.exp(-sigma_deltas)  # (batch_size, num_samples)

        # Comptue T_i = prod_{j=1}^{i-1} (1 - alpha_j)
        eps = 1e-10  # to prevent numerical issues
        ones = torch.ones(batch_size, 1, device=alphas.device, dtype=alphas.dtype)
        trans = torch.cumprod(torch.clamp(1.0 - alphas + eps, min=eps), dim=1)
        # add initial value of 1 to T
        trans = torch.cat([ones, trans[:, :-1]], dim=1)  # (batch_size, num_samples)

        weights = trans * alphas  # (batch_size, num_samples)

        rgb_pred = torch.sum(weights[..., None] * colors, dim=1)  # [B, 3]

        T_end = torch.prod(
            torch.clamp(1.0 - alphas + eps, min=eps), dim=1, keepdim=True
        )  # [B, 1]

        background_tensor = torch.tensor(
            bg_color, device=rgb_pred.device, dtype=rgb_pred.dtype
        )

        rgb_pred = rgb_pred + T_end * background_tensor  # [B, 3]

        return rgb_pred, weights
