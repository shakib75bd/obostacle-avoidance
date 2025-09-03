import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import timm
import urllib.request
import os
from pathlib import Path

class DepthEstimator:
    """
    MiDaS depth estimation with Monte Carlo dropout for uncertainty estimation
    """
    def __init__(self, model_type="MiDaS_small", device=None, num_samples=10, dropout_rate=0.2):
        """
        Initialize the depth estimator

        Args:
            model_type: MiDaS model version ("MiDaS_small" or "DPT_Hybrid")
            device: torch device (will use cuda if available when None)
            num_samples: Number of MC dropout samples for uncertainty
            dropout_rate: Dropout rate for uncertainty estimation
        """
        self.model_type = model_type
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate

        print(f"Using device: {self.device}")

        # Model paths
        self.model_urls = {
            "MiDaS_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
        }

        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

        self.model_path = self.model_dir / f"{model_type}.pt"

        # Download model if not exists
        if not self.model_path.exists():
            print(f"Downloading {model_type} model...")
            urllib.request.urlretrieve(self.model_urls[model_type], self.model_path)
            print(f"Model downloaded to {self.model_path}")

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)

        # Model-specific transforms
        if model_type == "MiDaS_small":
            self.img_size = (256, 256)  # Model input size
            self.net_w, self.net_h = 256, 256
        else:
            self.img_size = (384, 384)
            self.net_w, self.net_h = 384, 384

    def _load_model(self):
        """Load MiDaS model and patch with MC dropout"""
        if self.model_type == "MiDaS_small":
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False)
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)

            # Patch model with dropout for uncertainty estimation
            self._patch_model_with_dropout(model)

            return model
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _patch_model_with_dropout(self, model):
        """Add dropout layers to the model for uncertainty estimation"""
        # Enable dropout during inference
        def make_dropout_permanent(m):
            if isinstance(m, nn.Dropout):
                m.train()  # Keep dropout enabled during eval mode

        model.apply(make_dropout_permanent)

        # Add additional dropout layers for uncertainty estimation
        # This is a simplified approach - in a real implementation you might
        # want to add dropout at specific strategic locations in the network
        if hasattr(model, 'scratch'):
            if hasattr(model.scratch, 'output_conv'):
                # Add dropout before the final convolution
                orig_conv = model.scratch.output_conv
                new_seq = nn.Sequential(
                    nn.Dropout(p=self.dropout_rate),
                    orig_conv
                )
                model.scratch.output_conv = new_seq

    def _prepare_input(self, img):
        """Prepare image for the model"""
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB) / 255.0

        # Resize to model input size
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = (torch.from_numpy(img).permute(2, 0, 1) - mean[:, None, None]) / std[:, None, None]

        return img.unsqueeze(0).to(self.device).float()

    def _process_output(self, depth, original_size):
        """Process model output to usable depth map"""
        # Get the depth map
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=original_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

        # Convert to numpy
        depth = depth.cpu().numpy()

        # Normalize for visualization
        depth_min = depth.min()
        depth_max = depth.max()

        # Avoid division by zero
        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth)

        return depth_normalized[0]  # Return the first (and only) image

    def estimate_depth_with_uncertainty(self, img):
        """
        Estimate depth with uncertainty using Monte Carlo dropout

        Args:
            img: Input BGR image

        Returns:
            depth_map: Normalized depth map
            uncertainty_map: Uncertainty map (standard deviation across samples)
            colored_depth: Colorized depth map for visualization
        """
        h, w = img.shape[:2]
        original_size = (h, w)  # PyTorch expects (height, width) format
        input_tensor = self._prepare_input(img)

        # Run multiple forward passes with dropout enabled
        with torch.no_grad():
            self.model.eval()
            depth_samples = []

            for _ in range(self.num_samples):
                prediction = self.model(input_tensor)
                depth_samples.append(prediction)

            # Stack all samples
            depth_samples = torch.stack(depth_samples)

            # Calculate mean depth
            mean_depth = torch.mean(depth_samples, dim=0)

            # Calculate uncertainty (standard deviation)
            uncertainty = torch.std(depth_samples, dim=0)

            # Process the outputs
            depth_map = self._process_output(mean_depth, original_size)
            uncertainty_map = self._process_output(uncertainty, original_size)

            # Create colored depth map for visualization
            colored_depth = cv2.applyColorMap(
                np.clip((depth_map * 255), 0, 255).astype(np.uint8),
                cv2.COLORMAP_INFERNO
            )

            # Create colored uncertainty map
            colored_uncertainty = cv2.applyColorMap(
                np.clip((uncertainty_map * 255), 0, 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            return depth_map, uncertainty_map, colored_depth, colored_uncertainty
