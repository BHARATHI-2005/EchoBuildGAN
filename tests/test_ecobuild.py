import io
import base64
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from PIL import Image
import torch

from ecobuild.inference import (
    InvalidImageError,
    checkpoint_path,
    decode_image,
    generate_image,
    load_generator,
)
from ecobuild.image_provider import (
    DEFAULT_CF_MODEL,
    ImageProviderConfigurationError,
    generate_cloudflare_image,
    validate_cloudflare_account_id,
)
from ecobuild.reports import build_design_report
from models.networks import define_G


def encoded_image(image_format="PNG", size=(32, 24)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(40, 100, 60)).save(buffer, format=image_format)
    return buffer.getvalue()


class IdentityGenerator:
    def __call__(self, tensor):
        return tensor


class EcoBuildTests(unittest.TestCase):
    def test_decode_png_and_jpeg_as_rgb(self):
        for image_format in ("PNG", "JPEG"):
            with self.subTest(image_format=image_format):
                image = decode_image(encoded_image(image_format))
                self.assertEqual(image.mode, "RGB")

    def test_rejects_non_image_and_unsupported_format(self):
        with self.assertRaises(InvalidImageError):
            decode_image(b"not an image")
        with self.assertRaises(InvalidImageError):
            decode_image(encoded_image("GIF"))

    def test_inference_preprocesses_and_returns_rgb_image(self):
        source = decode_image(encoded_image())
        output = generate_image(source, IdentityGenerator(), "cpu")
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, (256, 256))

    def test_checkpoint_path_uses_repository_naming(self):
        self.assertEqual(
            checkpoint_path("checkpoints", "demo", "latest"),
            Path("checkpoints") / "demo" / "latest_net_G.pth",
        )

    def test_missing_checkpoint_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_generator(Path(directory) / "missing.pth", "cpu")

    def test_cloudflare_provider_requires_account_and_token(self):
        with self.assertRaisesRegex(
            ImageProviderConfigurationError, "ECOBUILD_CF_ACCOUNT_ID"
        ):
            generate_cloudflare_image(
                Image.new("RGB", (16, 16)),
                "Make an architectural rendering.",
                account_id="",
                token="",
            )

    def test_cloudflare_rejects_email_as_account_id_before_request(self):
        with patch("ecobuild.image_provider.urlopen") as urlopen:
            with self.assertRaisesRegex(
                ImageProviderConfigurationError,
                "32-character hexadecimal Cloudflare account ID",
            ):
                generate_cloudflare_image(
                    Image.new("RGB", (16, 16)),
                    "Make an architectural rendering.",
                    account_id="user@example.com",
                    token="test-token",
                )
        urlopen.assert_not_called()

    def test_cloudflare_accepts_32_character_hex_account_id(self):
        validate_cloudflare_account_id("0123456789abcdef0123456789abcdef")

    def test_cloudflare_provider_returns_generated_image(self):
        expected = Image.new("RGB", (32, 24), color=(100, 140, 80))
        encoded = io.BytesIO()
        expected.save(encoded, format="PNG")
        response_body = json.dumps(
            {
                "success": True,
                "result": {"image": base64.b64encode(encoded.getvalue()).decode()},
            }
        ).encode()
        with patch("ecobuild.image_provider.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = io.BytesIO(response_body)
            output = generate_cloudflare_image(
                Image.new("RGB", (2000, 1500)),
                "Render green architecture.",
                account_id="0123456789abcdef0123456789abcdef",
                token="test-token",
            )
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://api.cloudflare.com/client/v4/accounts/"
            "0123456789abcdef0123456789abcdef/"
            f"ai/run/{DEFAULT_CF_MODEL}",
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertIn(b'name="prompt"', request.data)
        self.assertIn(b'name="image"; filename="input.jpg"', request.data)
        input_bytes = request.data.split(
            b"Content-Type: image/jpeg\r\n\r\n", 1
        )[1].split(b"\r\n--", 1)[0]
        with Image.open(io.BytesIO(input_bytes)) as sent_image:
            self.assertEqual(sent_image.size, (1024, 768))
        self.assertEqual(output.size, (32, 24))

    def test_cloudflare_404_explains_account_id_and_provider_error(self):
        response = io.BytesIO(
            json.dumps(
                {
                    "errors": [
                        {"code": 7003, "message": "Model not available"}
                    ]
                }
            ).encode()
        )
        error = HTTPError(
            "https://api.cloudflare.com/client/v4/accounts/example/ai/run/test",
            404,
            "Not Found",
            {},
            response,
        )
        with patch("ecobuild.image_provider.urlopen", side_effect=error):
            with self.assertRaisesRegex(
                RuntimeError,
                "HTTP 404.*error 7003: Model not available",
            ):
                generate_cloudflare_image(
                    Image.new("RGB", (16, 16)),
                    "Make an architectural rendering.",
                    account_id="0123456789abcdef0123456789abcdef",
                    token="test-token",
                )

    def test_loads_compatible_pix2pix_generator_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest_net_G.pth"
            generator = define_G(
                3, 3, 1, "unet_256", norm="batch", use_dropout=True
            )
            torch.save(generator.state_dict(), checkpoint)
            loaded = load_generator(
                checkpoint,
                "cpu",
                net_g="unet_256",
                ngf=1,
                norm="batch",
                use_dropout=True,
            )
            self.assertFalse(loaded.training)
            generated = generate_image(
                decode_image(encoded_image()), loaded, "cpu"
            )
            self.assertEqual(generated.size, (256, 256))

    def test_report_contains_selected_facts_and_caveats(self):
        report = build_design_report(
            description="Small courtyard home",
            preferences=["Green courtyards"],
            original_filename="sketch.png",
            generated=False,
        )
        self.assertIn("Small courtyard home", report)
        self.assertIn("Green courtyards", report)
        self.assertIn("sketch.png", report)
        self.assertIn("No generated image available", report)
        self.assertIn("not a verified", report)


if __name__ == "__main__":
    unittest.main()
