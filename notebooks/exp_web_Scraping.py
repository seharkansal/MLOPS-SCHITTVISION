import requests
from time import sleep
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                  ' Chrome/90.0.4430.212 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

def get_with_retries(url, retries=5, backoff=1):
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Request failed ({i+1}/{retries}): {e}")
            time.sleep(backoff * (2 ** i))  # exponential backoff
    raise Exception("Failed to fetch the URL after retries")

url = 'https://transcripts.foreverdreaming.org/viewtopic.php?t=16052'
html = get_with_retries(url)
print("Page fetched successfully!")