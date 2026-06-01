import time
import requests
from pathlib import Path
from datetime import datetime

API_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/camsys%2Fall-alerts"

SAVE_DIR = Path("data/raw_service_alerts")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

while True:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    response = requests.get(API_URL, timeout=30)
    response.raise_for_status()

    file_path = SAVE_DIR / f"alerts_{timestamp}.pb"

    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"saved {file_path}")

    time.sleep(300)  # every 5 minutes