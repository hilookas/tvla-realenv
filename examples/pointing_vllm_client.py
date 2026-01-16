import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI


class VLLMOnlineClient:
    """Client for calling vLLM online API with support for images and videos"""

    def __init__(
        self,
        base_url: str = "http://localhost:22002/v1",
        api_key: str = "EMPTY",
        model_name: str = "Embodied-R1.5-SFT-v1",
    ):
        """
        Initialize vLLM online client

        Args:
            base_url: Base URL of the vLLM server
            api_key: API key (use "EMPTY" if no authentication)
            model_name: Model name as specified in --served-model-name
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    @staticmethod
    def encode_image(image_path: Union[str, Path]) -> str:
        """
        Encode image to base64 string

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string with data URI prefix
        """
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')

        # Detect image format
        image_path = Path(image_path)
        suffix = image_path.suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(suffix, 'image/jpeg')

        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def encode_video(video_path: Union[str, Path]) -> str:
        """
        Encode video to base64 string

        Args:
            video_path: Path to the video file

        Returns:
            Base64 encoded video string with data URI prefix
        """
        with open(video_path, "rb") as video_file:
            encoded = base64.b64encode(video_file.read()).decode('utf-8')

        # Detect video format
        video_path = Path(video_path)
        suffix = video_path.suffix.lower()
        mime_type = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm'
        }.get(suffix, 'video/mp4')

        return f"data:{mime_type};base64,{encoded}"

    def prepare_messages_from_test_case(
        self,
        test_case: Dict[str, Any],
        base_path: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Build messages from test case data (similar to offline example)

        Args:
            test_case: Test case dictionary with keys: type, prompt, image/video
            base_path: Base path for relative file paths

        Returns:
            Formatted messages for OpenAI API
        """
        content = []
        prompt_text = test_case["prompt"]
        test_type = test_case["type"]

        if test_type == "single_image":
            # Single image case
            image_path = Path(base_path) / test_case["image"]
            encoded_image = self.encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": encoded_image}
            })
            content.append({"type": "text", "text": prompt_text})

        elif test_type == "multi_image":
            # Multiple images case
            for img_path in test_case["image"]:
                full_path = Path(base_path) / img_path
                encoded_image = self.encode_image(full_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": encoded_image}
                })
            content.append({"type": "text", "text": prompt_text})

        elif test_type == "video":
            # Video case - use video_url type (vLLM specific)
            video_data = test_case["video"]
            if isinstance(video_data[0], list):
                raise ValueError("Unsupport type!")
            else:
                # Video file - use video_url type
                video_path = Path(base_path) / video_data[0]
                encoded_video = self.encode_video(video_path)
                content.append({
                    "type": "video_url",  # Use video_url for video files
                    "video_url": {"url": encoded_video}
                })
            content.append({"type": "text", "text": prompt_text})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        return messages
