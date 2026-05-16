"""Vision utilities for multimodal message construction."""

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# Supported image MIME types
SUPPORTED_IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


def encode_image_base64(file_path: Union[str, Path]) -> Tuple[str, str]:
    """Read an image file and return (base64_data, mime_type).

    Args:
        file_path: Path to the image file.

    Returns:
        Tuple of (base64_encoded_data, mime_type).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not a supported image format.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    suffix = path.suffix.lower()
    mime_type = SUPPORTED_IMAGE_TYPES.get(suffix)
    if mime_type is None:
        # Fall back to mimetypes module
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None or not mime_type.startswith("image/"):
            raise ValueError(
                f"Unsupported image format: {suffix}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_TYPES.keys()))}"
            )

    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, mime_type


def image_to_content_part(
    source: Union[str, Path],
    mime_type: Optional[str] = None,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert an image source to an OpenAI-format content part dict.

    Args:
        source: Either a file path, a URL (starting with http:// or https://),
                or raw base64 string. If a file path, the image is read and
                base64-encoded automatically.
        mime_type: MIME type of the image. Required when source is raw base64.
                   Auto-detected when source is a file path. Ignored for URLs.
        detail: Optional detail level ("auto", "low", "high"). Only used by
                providers that support it (e.g. OpenAI).

    Returns:
        A dict in OpenAI content-part format:
        {"type": "image_url", "image_url": {"url": "data:...;base64,..."}}

    Raises:
        FileNotFoundError: If source is a file path and file does not exist.
        ValueError: If source is base64 but mime_type is not provided.
    """
    source_str = str(source)

    if source_str.startswith("http://") or source_str.startswith("https://"):
        # URL-based image
        image_url: Dict[str, Any] = {"url": source_str}
        if detail:
            image_url["detail"] = detail
        return {"type": "image_url", "image_url": image_url}

    # Check if it's a file path (exists on disk)
    path = Path(source_str)
    if path.exists():
        b64_data, detected_mime = encode_image_base64(path)
        mime_type = mime_type or detected_mime
        url = f"data:{mime_type};base64,{b64_data}"
        image_url = {"url": url}
        if detail:
            image_url["detail"] = detail
        return {"type": "image_url", "image_url": image_url}

    # Assume raw base64 string
    if mime_type is None:
        raise ValueError(
            "mime_type is required when source is a base64 string. "
            "Provide mime_type='image/jpeg' or similar."
        )
    url = f"data:{mime_type};base64,{source_str}"
    image_url = {"url": url}
    if detail:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def create_image_message(
    image_source: Union[str, Path],
    prompt: str = "Describe this image.",
    mime_type: Optional[str] = None,
    detail: Optional[str] = None,
    role: str = "user",
) -> Dict[str, Any]:
    """Create a complete message dict with image and text, ready for chat_complete().

    Args:
        image_source: File path, URL, or base64 string.
        prompt: Text prompt to accompany the image.
        mime_type: MIME type (required for base64 strings, auto-detected for files).
        detail: Optional detail level for the image.
        role: Message role (default "user").

    Returns:
        A message dict with content array in OpenAI format:
        {"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}
    """
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt},
        image_to_content_part(image_source, mime_type=mime_type, detail=detail),
    ]
    return {"role": role, "content": content}
