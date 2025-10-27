#!/usr/bin/env python3

import requests
import json
import time

def test_api():
    base_url = "http://localhost:5000"
    
    print("Testing Face Mask Detection API...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.text}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test current status endpoint
    try:
        response = requests.get(f"{base_url}/api/current-status", timeout=5)
        print(f"Current status: {response.status_code}")
        if response.status_code == 200:
            print(f"Status response: {response.text}")
    except Exception as e:
        print(f"Current status failed: {e}")
    
    # Test alerts endpoint
    try:
        response = requests.get(f"{base_url}/api/alerts", timeout=5)
        print(f"Alerts: {response.status_code}")
        if response.status_code == 200:
            print(f"Alerts response: {response.text}")
    except Exception as e:
        print(f"Alerts failed: {e}")
    
    # Test stats endpoint
    try:
        response = requests.get(f"{base_url}/api/stats", timeout=5)
        print(f"Stats: {response.status_code}")
        if response.status_code == 200:
            print(f"Stats response: {response.text}")
    except Exception as e:
        print(f"Stats failed: {e}")

if __name__ == "__main__":
    test_api()
