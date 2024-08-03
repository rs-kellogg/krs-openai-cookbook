from typing import Dict
from pandas.core.frame import DataFrame
import pytest
import openai
import openaihelper.functions as F
import os
import json


def test_config(config: Dict):
    keys = [
        "system_prompt",
        "user_prompt",
        "encoding_name",
        "model_name",
        "max_token_len",
    ]
    for key in keys:
        assert key in config


def test_sample1(sample_1: DataFrame):
    assert len(sample_1) == 1


def test_sample1(sample_10: DataFrame):
    assert len(sample_10) == 10


def test_sample100(sample_100: DataFrame):
    assert len(sample_100) == 100


def test_count_tokens(config: Dict, sample_100: DataFrame):
    text = "This is a test message."
    encoding_name = "cl100k_base"
    num_tokens = F.count_tokens(text, encoding_name)
    assert num_tokens == 6
    
    texts = list(sample_100["text"])
    for idx, text in enumerate(texts):
        num_tokens = F.count_tokens(text, encoding_name)
        assert num_tokens > 0, f"Token length is zero for text at index {idx}"
        assert num_tokens <= config["max_token_len"], "Token length exceeds max token length"


@pytest.mark.skipif(
    "not config.getoption('--use-api')",
    reason="Only run when --use-api is given",
)
def test_chat_complete_success_mode(config: Dict, sample_100: DataFrame):
    system_prompt = config["system_prompt"]
    user_prompt = config["user_prompt"]
    model_name = config["model_name"]
    authors = list(sample_100["authors"])[0]
    text = list(sample_100["text"])[0]
    client = openai.OpenAI()
    response = F.chat_complete(
        client,
        text=text, 
        model_name=model_name, 
        system_prompt=system_prompt, 
        user_prompt=user_prompt
    )
    assert type(response) == str
    data = json.loads(response)
    print(data)
    assert "title" in data, "title not found"
    assert data['title'].lower() == (list(sample_100["title"])[0]).lower(), "title does not match"
    assert "authors" in data, "authors not found"
    for auth in data['authors']:
        assert "name" in auth, "name not found"
        assert auth['name'] in authors
        assert "email" in auth, "email not found"
        assert "affiliations" in auth, "affiliations reference not found"
    assert "affiliations" in data, "affiliations not found"
    for aff in data['affiliations']:
        assert "index" in aff, "index not found"
        assert "name" in aff, "name not found"
        assert "longitude" in aff, "longitude not found"
        assert "latitude" in aff, "latitude not found"

