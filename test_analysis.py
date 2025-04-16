#!/usr/bin/env python
"""
Test script for the market-ta-generator analysis endpoint.
"""

import json
import sys

import requests

# URL of the API endpoint
API_URL = "http://localhost:8000/api/v1/analyze"


def test_analysis(pair):
    """
    Test the analysis endpoint with the given trading pair.

    Args:
        pair: Trading pair to analyze (e.g., "eth_usdt")
    """
    # Create the request payload
    payload = {"pair": pair}

    print(f"\nTesting analysis for pair: {pair}")
    print("Sending request to:", API_URL)
    print("Request payload:", json.dumps(payload, indent=2))

    try:
        # Send the request to the API
        response = requests.post(API_URL, json=payload)

        # Print response status code
        print(f"Response status code: {response.status_code}")

        # Parse and print the response
        if response.status_code == 200:
            response_data = response.json()
            print("\nResponse:")
            print(json.dumps(response_data, indent=2))

            # Print the analysis separately for better readability
            if response_data.get("status") == "success" and response_data.get(
                "analysis"
            ):
                print("\n=== Analysis Result ===")
                print(response_data["analysis"])
                print("=====================")
            else:
                print(
                    "\nError in response:",
                    response_data.get("message", "Unknown error"),
                )
        else:
            print("\nError:", response.text)

    except Exception as e:
        print(f"Error sending request: {str(e)}")


if __name__ == "__main__":
    # Use command line argument or default to "eth_usdt"
    pair = sys.argv[1] if len(sys.argv) > 1 else "eth_usdt"
    test_analysis(pair)
