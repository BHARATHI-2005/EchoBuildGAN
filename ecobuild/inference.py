from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageOps

from data.base_dataset import get_transform
from models.networks import define_G


SUPPORTED_FORMATS = {"PNG", "JPEG"}
MAX_IMAGE_PIXELS = 25_000_000
IMAGE_SIZE = 256


class InvalidImageError(ValueError):
    """Raised when an uploaded file is not a supported, decodable image."""


class CheckpointError(RuntimeError):
    """Raised when a configured generator checkpoint cannot be loaded."""


def decode_image(image_bytes: bytes) -> Image.Image:
    """Validate an uploaded PNG/JPEG and return an EXIF-corrected RGB copy."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in SUPPORTED_FORMATS:
                raise InvalidImageError("Please upload a PNG or JPEG image.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise InvalidImageError(
                    "This image is too large to process. Please use an image "
                    "under 25 megapixels."
                )
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except InvalidImageError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise InvalidImageError(
            "The uploaded file could not be decoded as a valid image."
        ) from error


def checkpoint_path(
    checkpoints_dir: str | Path, model_name: str, epoch: str
) -> Path:
    """Return the standard checkpoint path used by this repository."""
    return Path(checkpoints_dir) / model_name / f"{epoch}_net_G.pth"


def load_generator(
    checkpoint: str | Path,
    device: str | torch.device,
    *,
    net_g: str = "unet_256",
    ngf: int = 64,
    norm: str = "batch",
    use_dropout: bool = True,
) -> torch.nn.Module:
    """Load a strict Pix2Pix generator state dict on the selected device."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint}")

    generator = define_G(
        input_nc=3,
        output_nc=3,
        ngf=ngf,
        netG=net_g,
        norm=norm,
        use_dropout=use_dropout,
    )
    try:
        state_dict = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        if not isinstance(state_dict, dict):
            raise CheckpointError(
                "The checkpoint is not a Pix2Pix generator state dictionary."
            )
        generator.load_state_dict(state_dict, strict=True)
    except CheckpointError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise CheckpointError(
            "The checkpoint is incompatible with the configured Pix2Pix "
            "generator. Check its architecture, normalization, filter count, "
            "dropout setting, and that it is a generator checkpoint."
        ) from error

    generator.to(torch.device(device))
    generator.eval()
    return generator


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Apply the repository's deterministic Pix2Pix test transform."""
    options = SimpleNamespace(
        preprocess="resize_and_crop",
        load_size=IMAGE_SIZE,
        crop_size=IMAGE_SIZE,
        no_flip=True,
    )
    transform = get_transform(options)
    return transform(image.convert("RGB"))


def generate_image(
    image: Image.Image,
    generator: torch.nn.Module,
    device: str | torch.device,
) -> Image.Image:
    """Run one deterministic generator pass and convert tanh output to RGB."""
    input_tensor = preprocess_image(image).unsqueeze(0).to(torch.device(device))
    with torch.inference_mode():
        generated = generator(input_tensor)

    if not isinstance(generated, torch.Tensor) or generated.ndim != 4:
        raise CheckpointError("The generator returned an invalid image tensor.")
    if generated.shape[0] != 1 or generated.shape[1] != 3:
        raise CheckpointError(
            "The generator must return one three-channel image."
        )

    pixels = (
        generated[0]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(np.ascontiguousarray(pixels))
