from pathlib import Path
from tqdm import tqdm
import pandas as pd
import openai
import json
import yaml
import tiktoken
from typing import Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


# -----------------------------------------------------------------------------
def config(config_file: Path) -> Dict:
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        return conf


# -----------------------------------------------------------------------------
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs) -> openai.types.chat.chat_completion.ChatCompletion:
    client = openai.OpenAI()
    response = client.chat.completions.create(**kwargs)
    return response


# -----------------------------------------------------------------------------
def chat_complete(
    client,
    model_name: str,
    user_prompt: str,
    system_prompt: str,
    text: str,
    temperature=0,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    seed=42,
) -> str:
    assert text is not None and len(text) > 0
    assert model_name is not None and len(model_name) > 0
    assert user_prompt is not None and len(user_prompt) > 0

    try:
        response = completion_with_backoff(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + '\n\n"""' + text + '\n\n"""'},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )
        return json.dumps(json.loads(response.choices[0].message.content))
    except Exception as e:
        return str(f"{response}")


# -----------------------------------------------------------------------------
def count_tokens(text: str, encoding_name: str) -> int:
    assert text is not None and len(text) > 0
    assert encoding_name is not None and len(encoding_name) > 0

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


# -----------------------------------------------------------------------------
def validate_result(result: str) -> bool:
    assert result is not None
    return True


# -----------------------------------------------------------------------------
def text2speech(client, text: str, model: str = "tts-1", voice: str = "alloy"):
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
    )
    return response


# -----------------------------------------------------------------------------
def speech2text(client, audio_file: str, model: str = "whisper-1"):
    response = client.audio.transcriptions.create(
        model=model,
        file=audio_file,
    )
    return response
