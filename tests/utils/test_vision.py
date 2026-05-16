"""Tests for vision utilities."""

import base64
import tempfile
from pathlib import Path

import pytest

from esperanto.utils.vision import (
    create_image_message,
    encode_image_base64,
    image_to_content_part,
)


@pytest.fixture
def sample_png(tmp_path):
    """Create a minimal valid PNG file for testing."""
    # Minimal 1x1 transparent PNG
    png_data = (
        b'\x89PNG\r\n\x1a\n'  # PNG signature
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
        b'\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
        b'\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    path = tmp_path / "test.png"
    path.write_bytes(png_data)
    return path


@pytest.fixture
def sample_jpeg(tmp_path):
    """Create a minimal JPEG file for testing."""
    # Minimal JPEG (just markers)
    jpeg_data = b'\xff\xd8\xff\xe0' + b'\x00' * 20 + b'\xff\xd9'
    path = tmp_path / "test.jpg"
    path.write_bytes(jpeg_data)
    return path


class TestEncodeImageBase64:
    def test_encode_png(self, sample_png):
        data, mime = encode_image_base64(sample_png)
        assert isinstance(data, str)
        assert mime == "image/png"
        # Verify it's valid base64
        decoded = base64.standard_b64decode(data)
        assert decoded[:4] == b'\x89PNG'

    def test_encode_jpeg(self, sample_jpeg):
        data, mime = encode_image_base64(sample_jpeg)
        assert isinstance(data, str)
        assert mime == "image/jpeg"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            encode_image_base64("/nonexistent/path/image.png")

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Unsupported image format"):
            encode_image_base64(path)

    def test_accepts_string_path(self, sample_png):
        data, mime = encode_image_base64(str(sample_png))
        assert mime == "image/png"


class TestImageToContentPart:
    def test_from_file(self, sample_png):
        part = image_to_content_part(sample_png)
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_from_url(self):
        url = "https://example.com/image.jpg"
        part = image_to_content_part(url)
        assert part["type"] == "image_url"
        assert part["image_url"]["url"] == url

    def test_from_http_url(self):
        url = "http://example.com/image.jpg"
        part = image_to_content_part(url)
        assert part["image_url"]["url"] == url

    def test_from_base64_with_mime(self):
        b64 = base64.standard_b64encode(b"fake image data").decode()
        part = image_to_content_part(b64, mime_type="image/jpeg")
        assert part["type"] == "image_url"
        assert part["image_url"]["url"] == f"data:image/jpeg;base64,{b64}"

    def test_from_base64_without_mime_raises(self):
        b64 = base64.standard_b64encode(b"fake").decode()
        with pytest.raises(ValueError, match="mime_type is required"):
            image_to_content_part(b64)

    def test_detail_parameter(self, sample_png):
        part = image_to_content_part(sample_png, detail="high")
        assert part["image_url"]["detail"] == "high"

    def test_detail_with_url(self):
        part = image_to_content_part("https://example.com/img.png", detail="low")
        assert part["image_url"]["detail"] == "low"

    def test_no_detail_by_default(self, sample_png):
        part = image_to_content_part(sample_png)
        assert "detail" not in part["image_url"]


class TestCreateImageMessage:
    def test_basic_message(self, sample_png):
        msg = create_image_message(sample_png, "Describe this.")
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0] == {"type": "text", "text": "Describe this."}
        assert msg["content"][1]["type"] == "image_url"

    def test_default_prompt(self, sample_png):
        msg = create_image_message(sample_png)
        assert msg["content"][0]["text"] == "Describe this image."

    def test_url_source(self):
        msg = create_image_message("https://example.com/img.jpg", "What is this?")
        assert msg["content"][1]["image_url"]["url"] == "https://example.com/img.jpg"

    def test_custom_role(self, sample_png):
        msg = create_image_message(sample_png, role="assistant")
        assert msg["role"] == "assistant"
