#!/usr/bin/env python3
"""Read turns.bin and print the turn array values"""
import os

# Read turns.bin as binary stream
turns_bin_path = os.path.join(os.path.dirname(__file__), '..', 'turns.bin')

try:
    with open(turns_bin_path, 'rb') as fh:
        turn_array = list(fh.read())
    
    print(f"Turn array read from {turns_bin_path}:")
    print(f"Length: {len(turn_array)}")
    print(f"Values: {turn_array}")
    
    # Pretty print with names
    turn_names = {0: 'Right', 1: 'Straight', 2: 'Left', 3: 'U-turn'}
    print("\nTurns (readable):")
    for i, turn_type in enumerate(turn_array):
        label = turn_names.get(turn_type, f'Unknown({turn_type})')
        print(f"  Turn {i}: {turn_type} ({label})")

except FileNotFoundError:
    print(f"Error: {turns_bin_path} not found")
except Exception as e:
    print(f"Error reading file: {e}")
