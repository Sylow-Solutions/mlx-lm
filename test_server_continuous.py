#!/usr/bin/env python3
import json
import random
import time

import requests

url = "http://192.168.4.52:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# Different types of prompts to test
prompts = [
    # Simple greetings
    "Hello!",
    "Hi there, how are you?",
    # Questions
    "What is the capital of France?",
    "Can you explain what machine learning is?",
    "What's 2 + 2?",
    # Tasks
    "Write a haiku about coding",
    "Give me 3 tips for learning Python",
    "Translate 'hello world' to Spanish",
    # Longer prompts
    "Tell me a short story about a robot who learns to paint",
    "Explain the difference between a list and a tuple in Python",
    # Code-related
    "Write a Python function to reverse a string",
    "What's the time complexity of bubble sort?",
    # Conversational
    "What's your favorite programming language and why?",
    "Can you help me debug my code?",
    # Edge cases
    "🚀🌟",
    "HELLO IN ALL CAPS",
    "1234567890",
    "```python\nprint('hello')\n```",
]

print(
    "Starting continuous requests every 2 seconds with varied prompts. Press Ctrl+C to stop."
)

request_count = 0
while True:
    try:
        request_count += 1

        # Select a random prompt
        prompt = random.choice(prompts)

        data = {
            "model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }

        print(f"\n[Request #{request_count}]")
        print(f"Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"Response: {content[:100]}{'...' if len(content) > 100 else ''}")
            print(f"Tokens used: {result['usage']['total_tokens']}")
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")

        time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\nStopped after {request_count} requests.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(2)
