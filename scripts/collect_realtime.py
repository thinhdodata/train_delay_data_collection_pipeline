import time
import requests
from datetime import datetime
from pathlib import Path

API_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs"

SAVE_DIR = Path("data/raw_snapshots")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

while True:
    response = requests.get(API_URL)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_path = SAVE_DIR / f"snapshot_{timestamp}.pb"

    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"saved {file_path}")

    time.sleep(30)