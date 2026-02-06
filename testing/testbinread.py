import asyncio
import time
import websockets
import json
import base64
import numpy as np
import cv2
import sys

turn_array = []

def load_turns_from_file(filepath):
    """
    Read turn sequence from a binary file.
    Each byte represents one turn (0=right, 1=forward, 2=left, 3=u-turn).
    Returns a list of turn values.
    """
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        turns = list(data)
        print(f"Loaded {len(turns)} turns from {filepath}")
        return turns
    except FileNotFoundError:
        print(f"Error: File {filepath} not found. Using empty turn array.")
        return []
    except Exception as e:
        print(f"Error reading turns file: {e}")
        return []

# Load turns from binary file if it exists
TURNS_FILE = f"turns.bin"

def main():
    global turn_array
    turn_array = load_turns_from_file(TURNS_FILE)
    print(f"Turn array: {turn_array}")
    print(f"First turn: {turn_array[len(turn_array)-1]}")
    # turn_array.pop()
    # print(f"Turn array: {turn_array}")
    # if len(turn_array) > 0:
    #     print(f"First turn: {turn_array[len(turn_array)-1]}")

if __name__ == "__main__":
    main()