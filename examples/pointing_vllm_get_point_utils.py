from pointing_vllm_client import VLLMOnlineClient
from pointing_vllm_point_utils import omni_decode_points
from PIL import Image
import cv2
import numpy as np

def get_point_vllm(image_rgb, text_prompt="you need to grasp the mug", save_path="debug.png"):
    # Model and server configuration
    base_url = "http://172.28.102.11:22002/v1"
    api_key = "EMPTY"
    model_name = "Embodied-R1.5-SFT-v1"

    # Sampling parameters
    greedy = False
    seed = 3407
    top_p = 0.8
    top_k = 20
    temperature = 0.7
    repetition_penalty = 1.0
    presence_penalty = 1.5
    max_tokens = 4096  # out_seq_length

    # Video processing parameters (for mm_processor_kwargs)
    video_fps = 2  # Frames per second for video sampling
    video_do_sample_frames = True  # Enable frame sampling

    # Initialize client
    client = VLLMOnlineClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name
    )

    height, width = image_rgb.shape[:2]

    tmp_image_path = "vllm_image.png"

    Image.fromarray(image_rgb).save(tmp_image_path)

    test_case = {
        'idx': 2,
        'answer': '',
        'prompt': 'Provide one or more points coordinate of objects region this sentence describes: ' + text_prompt + '. The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].',
        'image': tmp_image_path,
        'video': '',
        'type': 'single_image'
    }

    print("-"*20 + "prompt")
    print(test_case["prompt"])
    print("-"*20 + "image_path")
    print(test_case["image"]) # image path

    # Build messages for this test case
    messages = client.prepare_messages_from_test_case(test_case)

    # Generate output
    # Use mm_processor_kwargs for video processing (safe to use)
    # Note: Do NOT add top_k, repetition_penalty, presence_penalty - they cause crashes
    response = client.client.chat.completions.create(
        model=client.model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        extra_body={
            "mm_processor_kwargs": {
                "fps": video_fps,
                "do_sample_frames": video_do_sample_frames
            }
        }
    )
    generated_text = response.choices[0].message.content

    print("-"*20 + "generated")
    print(generated_text)

    import numpy as np

    pointing = (np.array(omni_decode_points(generated_text)) / 1000 * np.array([width, height]))[0]

    if save_path:
        img = cv2.imread(test_case["image"])
        img = cv2.circle(img, (int(pointing[0]), int(pointing[1])), 5, (0,0,255), -1)
        cv2.imwrite(save_path, img)

    return pointing # x, y following opencv camera coord


if __name__ == "__main__":
    print(get_point_vllm(np.array(Image.open("aff.png")), "you need to grasp the mug"))