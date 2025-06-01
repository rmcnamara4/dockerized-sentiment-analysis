import requests

url = "http://127.0.0.1:8000/predict"

def test_requests(data):
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # raises error if not 200
        print("Response:", response.json())
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error: {err}")
    except Exception as e:
        print(f"Other error: {e}")

if __name__ == "__main__":
    test_data = {
        "texts": [
            "I love this!",
            "This is terrible."
        ]
    }
    test_requests(test_data)
