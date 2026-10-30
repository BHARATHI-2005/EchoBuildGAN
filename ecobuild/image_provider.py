from __future__ import annotations

import base64
import json
import re
import uuid
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


DEFAULT_CF_MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
MAX_REMOTE_IMAGE_SIZE = (1024, 1024)
REQUEST_TIMEOUT = 180


class ImageProviderConfigurationError(RuntimeError):
    """Raised when the hosted image provider is not configured."""


class ImageProviderRequestError(RuntimeError):
    """Raised when hosted image generation fails."""


def validate_cloudflare_account_id(account_id: str) -> None:
    if not re.fullmatch(r"[a-fA-F0-9]{32}", account_id.strip()):
        raise ImageProviderConfigurationError(
            "ECOBUILD_CF_ACCOUNT_ID must be your 32-character hexadecimal "
            "Cloudflare account ID, not your email address. Find it in the "
            "Cloudflare dashboard under Workers AI or Account Home."
        )


def _multipart_body(image: Image.Image, prompt: str) -> tuple[bytes, str]:
    boundary = f"----EchoBuildGAN{uuid.uuid4().hex}"
    source = image.convert("RGB")
    source.thumbnail(MAX_REMOTE_IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_buffer = BytesIO()
    source.save(image_buffer, format="JPEG", quality=90)

    body = b"".join(
        (
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="prompt"\r\n\r\n'
                f"{prompt}\r\n"
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="image"; filename="input.jpg"\r\n'
                "Content-Type: image/jpeg\r\n\r\n"
            ).encode("utf-8"),
            image_buffer.getvalue(),
            f"\r\n--{boundary}--\r\n".encode("ascii"),
        )
    )
    return body, f"multipart/form-data; boundary={boundary}"


def generate_cloudflare_image(
    image: Image.Image,
    prompt: str,
    *,
    account_id: str,
    token: str,
    model: str = DEFAULT_CF_MODEL,
) -> Image.Image:
    """Edit an architectural image with Cloudflare Workers AI."""
    if not account_id.strip():
        raise ImageProviderConfigurationError(
            "Set ECOBUILD_CF_ACCOUNT_ID in your local .env file."
        )
    validate_cloudflare_account_id(account_id)
    if not token.strip():
        raise ImageProviderConfigurationError(
            "Set ECOBUILD_CF_API_TOKEN in your local .env file."
        )
    if not model.strip():
        raise ImageProviderConfigurationError(
            "Set ECOBUILD_CF_MODEL to a valid Workers AI image model ID."
        )

    body, content_type = _multipart_body(image, prompt)
    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id.strip()}/ai/run/{model.strip()}"
    )
    request = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token.strip()}",
            "Content-Type": content_type,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read())
    except HTTPError as error:
        try:
            error_payload = json.loads(error.read())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            error_payload = {}
        errors = (
            error_payload.get("errors", [])
            if isinstance(error_payload, dict)
            else []
        )
        provider_detail = ""
        if errors and isinstance(errors[0], dict):
            code = errors[0].get("code")
            detail = errors[0].get("message")
            if isinstance(detail, str) and detail.strip():
                provider_detail = f" Cloudflare: {detail.strip()[:300]}"
                if code is not None:
                    provider_detail = (
                        f" Cloudflare error {code}: {detail.strip()[:300]}"
                    )

        if error.code == 401:
            message = (
                "Cloudflare rejected the API token. Check its Workers AI "
                "permissions and account ID."
            )
        elif error.code == 403:
            message = (
                "Cloudflare denied access to Workers AI. Check the token's "
                "Workers AI permissions and account ID."
            )
        elif error.code == 429:
            message = (
                "The Cloudflare Workers AI daily free allocation or rate "
                "limit was reached. Try again after the limit resets."
            )
        elif error.code == 402:
            message = (
                "Cloudflare Workers AI could not run this request. Check "
                "your account plan and the model's availability."
            )
        elif error.code == 404:
            message = (
                "Cloudflare returned HTTP 404. Verify that the configured "
                "Cloudflare account ID and model ID are correct."
                f"{provider_detail}"
            )
        else:
            message = (
                f"Cloudflare image generation failed (HTTP {error.code}). "
                "Check the account, model ID, token permissions, and Workers "
                f"AI availability.{provider_detail}"
            )
        raise ImageProviderRequestError(message) from error
    except (TimeoutError, URLError, OSError) as error:
        raise ImageProviderRequestError(
            "Could not reach Cloudflare Workers AI. Check your network and "
            "try again."
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ImageProviderRequestError(
            "Cloudflare returned an invalid response."
        ) from error

    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise ImageProviderRequestError(
            "Cloudflare Workers AI did not complete the image request. "
            "Check that the model is enabled for your account."
        )

    result = payload.get("result")
    encoded_image = result.get("image") if isinstance(result, dict) else None
    if not isinstance(encoded_image, str) or not encoded_image:
        raise ImageProviderRequestError(
            "Cloudflare did not return a valid image."
        )

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
        with Image.open(BytesIO(image_bytes)) as generated:
            generated.load()
            return generated.convert("RGB")
    except (ValueError, OSError) as error:
        raise ImageProviderRequestError(
            "Cloudflare returned an invalid image."
        ) from error
