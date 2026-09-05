import torch

class FusedFeatureSampler(torch.nn.Module):
    def __init__(self, output_dims=768, input_dims=[1536] * 4, num_views=2, sample_radius=3, use_embeddings=True):
        super(FusedFeatureSampler, self).__init__()
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.num_views = num_views
        self.sample_radius = sample_radius

        assert sample_radius % 2 == 1, "sample_radius must be an odd integer."

        if use_embeddings:
            # Only view embeddings, not level embeddings since we're fusing levels
            self.embeddings = torch.nn.Parameter(torch.randn(num_views, output_dims))

        # Projectors for each feature level to output_dims // num_levels
        proj_dim = output_dims // len(input_dims)
        self.projectors = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, proj_dim) for input_dim in input_dims
        ])
        
        # Final projection after concatenation
        self.final_projection = torch.nn.Linear(proj_dim * len(input_dims), output_dims)

    def precompute_projection(self, input_features):
        """Apply the per-level projectors to the full feature maps ONCE (the projection is
        the expensive, iteration-invariant part of forward). Returns list[level] of
        [view] projected maps (B, H, W, proj_dim). Pass to forward(features_are_projected=True)
        across rollout iterations to skip re-projecting. Project-then-slice == slice-then-
        project (Linear is per-pixel), so this is numerically identical to the in-forward path."""
        return [[self.projectors[level](input_features[level][v].permute(0, 2, 3, 1))
                 for v in range(self.num_views)]
                for level in range(len(input_features))]

    def forward(self, input_features, sample_locations, sample_radius=None, features_are_projected=False):
        """
        Sample features around the given 2D locations from multiple views and fuse features from different levels.
        Unlike FeatureSampler which concatenates all levels, this fuses them into a single feature per view.

        If features_are_projected=True, input_features[level][view] are already projected maps
        (B, H, W, proj_dim) from precompute_projection() and the per-level projectors are skipped.

        Args:
            input_features (list of torch.Tensor): List of feature maps from different levels and views #levels list of #views tuples of shape (B, C, H, W).
            sample_locations (torch.Tensor): Locations to sample features from, shape (B, N, 2), the 2D coordinates (x, y) in [0, 1].

        Returns:
            torch.Tensor: Sampled and fused features, shape (B, N, num_views * sample_radius^2, output_dims).
            torch.Tensor: Corresponding masks indicating valid samples, shape (B, N, num_views * sample_radius^2).
        """

        B, N, _ = sample_locations.shape
        output = torch.full(
            (B, N, self.num_views * (self.sample_radius ** 2), self.output_dims), float('nan'), device=sample_locations.device
        )

        # Generate grid offsets around each sample location
        radius = self.sample_radius // 2 if sample_radius is None else sample_radius // 2
        offsets = torch.stack(
            torch.meshgrid(
                torch.arange(-radius, radius + 1, device=sample_locations.device),
                torch.arange(-radius, radius + 1, device=sample_locations.device),
                indexing='xy'
            ), dim=-1
        ).reshape(-1, 2)  # (sample_radius^2, 2)

        for view in range(self.num_views):
            # Collect features from all levels for this view
            level_features = []
            valid_mask_accumulated = None

            for level, feat_map in enumerate(input_features):
                if features_are_projected:
                    proj_feat_map = feat_map[view]                       # (B, H, W, proj_dim), pre-projected
                    B_f, H_f, W_f = proj_feat_map.shape[:3]
                else:
                    B_f, _, H_f, W_f = feat_map[view].shape
                assert B_f == B, "Batch size mismatch between input features and sample locations."

                pixel_coords = torch.stack([
                    sample_locations[:, :, 0] * (W_f - 1),
                    sample_locations[:, :, 1] * (H_f - 1)
                ], dim=-1)  # (B, N, 2) scaled to feature map size

                if not features_are_projected:
                    # Project feature map to proj_dim (B, H, W, proj_dim)
                    proj_feat_map = self.projectors[level](feat_map[view].permute(0, 2, 3, 1))

                # Compute all sample positions for this batch (B, N, sample_radius^2, 2)
                sample_grid = pixel_coords.unsqueeze(2) + offsets.unsqueeze(0).unsqueeze(0)

                # Clamp to valid range and check which samples are in bounds
                sample_grid_x = torch.round(sample_grid[..., 0]).long()
                sample_grid_y = torch.round(sample_grid[..., 1]).long()

                valid_mask = (
                    (sample_grid_x >= 0) & (sample_grid_x < W_f) &
                    (sample_grid_y >= 0) & (sample_grid_y < H_f)
                )  # (B, N, sample_radius^2)

                # Clamp coordinates to valid range for safe indexing
                sample_grid_x = sample_grid_x.clamp(0, W_f - 1)
                sample_grid_y = sample_grid_y.clamp(0, H_f - 1)

                # Sample features at valid positions
                sampled_features = proj_feat_map[
                    torch.arange(B, device=sample_locations.device)[:, None, None],
                    sample_grid_y,
                    sample_grid_x,
                    :
                ]  # (B, N, sample_radius^2, proj_dim)

                # Mask out invalid samples
                sampled_features[~valid_mask] = 0.0  # Use 0 for concatenation

                # Collect features from this level
                level_features.append(sampled_features)
                
                # Accumulate valid mask (require all levels to be valid)
                if valid_mask_accumulated is None:
                    valid_mask_accumulated = valid_mask
                else:
                    valid_mask_accumulated = valid_mask_accumulated & valid_mask

            # Concatenate features from all levels
            concatenated_features = torch.cat(level_features, dim=-1)  # (B, N, sample_radius^2, proj_dim * num_levels)
            
            # Apply final projection
            fused_features = self.final_projection(concatenated_features)  # (B, N, sample_radius^2, output_dims)

            if hasattr(self, 'embeddings'):
                # Add learnable view embeddings
                fused_features = fused_features + self.embeddings[view]

            # Mask out invalid samples with NaN
            fused_features[~valid_mask_accumulated] = float('nan')

            # Compute index in output tensor
            token_start_idx = view * (self.sample_radius ** 2)
            token_end_idx = token_start_idx + (self.sample_radius ** 2)

            # Store in output tensor
            output[:, :, token_start_idx:token_end_idx, :] = fused_features

        # Compute mask from NaN values (True where valid, False where NaN)
        mask = ~torch.isnan(output).any(dim=-1)  # (B, N, num_tokens)

        # Replace NaN values with zeros
        output = torch.nan_to_num(output, nan=0.0)

        return output, mask


class FeatureSampler(torch.nn.Module):
    def __init__(self, output_dims=768, input_dims=[1536] * 4, num_views=2, sample_radius=3, use_embeddings=True):
        super(FeatureSampler, self).__init__()
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.num_views = num_views
        self.sample_radius = sample_radius

        assert sample_radius % 2 == 1, "sample_radius must be an odd integer."

        if use_embeddings:
            self.embeddings = torch.nn.Parameter(torch.randn(num_views, len(input_dims), output_dims))

        self.projectors = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dims) for input_dim in input_dims
        ])

    def forward(self, input_features, sample_locations):
        """
        Sample features around the given 2D locations from multiple views and project them to a common dimension. Also add learnable embeddings to indicate
        the view and feature level. The sampling is done in a grid around the given locations defined by sample_radius. If the sample locations is out of bounds or close 
        to the border, tokens are only sampled from valid locations, and a mask is returned indicating valid samples.

        Args:
            input_features (list of torch.Tensor): List of feature maps from different levels and views #levels list of #views tuples of shape (B, C, H, W).
            sample_locations (torch.Tensor): Locations to sample features from, shape (B, N, 2), the 2D coordinates (x, y) in [0, 1].

        Returns:
            torch.Tensor: Sampled and projected features, shape (B, N, num_views * num_levels * sample_radius^2, output_dims).
            torch.Tensor: Corresponding masks indicating valid samples, shape (B, N, num_views * num_levels * sample_radius^2).
        """

        B, N, _ = sample_locations.shape
        output = torch.full(
            (B, N, self.num_views * len(self.input_dims) * (self.sample_radius ** 2), self.output_dims), float('nan'), device=sample_locations.device
        )

        # Generate grid offsets around each sample location
        radius = self.sample_radius // 2
        offsets = torch.stack(
            torch.meshgrid(
                torch.arange(-radius, radius + 1, device=sample_locations.device),
                torch.arange(-radius, radius + 1, device=sample_locations.device),
                indexing='xy'
            ), dim=-1
        ).reshape(-1, 2)  # (sample_radius^2, 2)

        for view in range(self.num_views):
            for level, feat_map in enumerate(input_features):
                B_f, _, H_f, W_f = feat_map[view].shape
                assert B_f == B, "Batch size mismatch between input features and sample locations."

                pixel_coords = torch.stack([
                    sample_locations[:, :, 0] * (W_f - 1),
                    sample_locations[:, :, 1] * (H_f - 1)
                ], dim=-1)  # (B, N, 2) scaled to feature map size

                # Project feature map to output dimension (B, H, W, output_dims)
                proj_feat_map = self.projectors[level](feat_map[view].permute(0, 2, 3, 1))

                # Compute all sample positions for this batch (B, N, sample_radius^2, 2)
                sample_grid = pixel_coords.unsqueeze(2) + offsets.unsqueeze(0).unsqueeze(0)

                # Clamp to valid range and check which samples are in bounds
                sample_grid_x = torch.round(sample_grid[..., 0]).long()
                sample_grid_y = torch.round(sample_grid[..., 1]).long()

                valid_mask = (
                    (sample_grid_x >= 0) & (sample_grid_x < W_f) &
                    (sample_grid_y >= 0) & (sample_grid_y < H_f)
                )  # (B, N, sample_radius^2)

                # Clamp coordinates to valid range for safe indexing
                sample_grid_x = sample_grid_x.clamp(0, W_f - 1)
                sample_grid_y = sample_grid_y.clamp(0, H_f - 1)

                # Sample features at valid positions
                # proj_feat_map: (B, H, W, output_dims)
                sampled_features = proj_feat_map[
                    torch.arange(B, device=sample_locations.device)[:, None, None],
                    sample_grid_y,
                    sample_grid_x,
                    :
                ]  # (B, N, sample_radius^2, output_dims)

                if hasattr(self, 'embeddings'):
                    # Add learnable embeddings for view and level information
                    sampled_features = sampled_features + self.embeddings[view, level]

                # Mask out invalid samples
                sampled_features[~valid_mask] = float('nan')

                # Compute index in output tensor
                token_start_idx = (view * len(self.input_dims) + level) * (self.sample_radius ** 2)
                token_end_idx = token_start_idx + (self.sample_radius ** 2)

                # Store in output tensor
                output[:, :, token_start_idx:token_end_idx, :] = sampled_features

        # Compute mask from NaN values (True where valid, False where NaN)
        mask = ~torch.isnan(output).any(dim=-1)  # (B, N, num_tokens)

        # Replace NaN values with zeros
        output = torch.nan_to_num(output, nan=0.0)

        return output, mask
    
