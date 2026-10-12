import io
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


# =========================
# CONFIG
# =========================
API_KEY = ""
DRIVE_FOLDER_ID = "-uGIdNrroR6FK" 

NUM_IMAGES = int(input("How many images do you want? ").strip())

# Google Street View endpoints
META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMG_URL = "https://maps.googleapis.com/maps/api/streetview"

# Overpass endpoint
OVERPASS_URL = "https://z.overpass-api.de/api/interpreter"

# Drive API scope
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# Australia-focused region boxes: (name, south, west, north, east)
REGIONS = [
     ("brisbane", -27.65, 152.85, -27.20, 153.30),
     ("gold_coast", -28.25, 153.20, -27.75, 153.50),
]

HEADINGS = [0, 90, 180, 270]

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 road-dataset-script/1.0"})


# =========================
# GOOGLE DRIVE AUTH + UPLOAD
# =========================
def get_drive_service():
    """
    Authenticate with Google Drive using OAuth desktop flow.
    Saves token.json after first successful login.
    """
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError(
                    "credentials.json not found. Download your Google OAuth desktop credentials "
                    "from Google Cloud and place it beside this script."
                )
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def upload_bytes_to_drive(service, data_bytes, filename, folder_id=None, mime_type="image/jpeg"):
    """
    Upload in-memory bytes directly to Google Drive.
    """
    metadata = {"name": filename}
    if folder_id:
        metadata["parents"] = [folder_id]

    media = MediaIoBaseUpload(io.BytesIO(data_bytes), mimetype=mime_type, resumable=True)

    uploaded = service.files().create(
        body=metadata,
        media_body=media,
        fields="id,name,webViewLink"
    ).execute()

    return uploaded


# =========================
# OSM ROAD SAMPLING
# =========================
def overpass_query_for_roads(region, max_results=100):
    """
    Query drivable roads from OSM Overpass within the region.
    """
    name, south, west, north, east = region

    query = f"""
    [out:json][timeout:30];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"]
         ["access"!="private"]
         ({south},{west},{north},{east});
    );
    out geom {max_results};
    """

    response = session.post(OVERPASS_URL, data=query.encode("utf-8"), timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("elements", [])


def random_point_on_way(way):
    """
    Pick a random interpolated point along one OSM road segment.
    """
    geom = way.get("geometry", [])
    if len(geom) < 2:
        return None

    i = random.randint(0, len(geom) - 2)
    a = geom[i]
    b = geom[i + 1]
    t = random.random()

    lat = a["lat"] + t * (b["lat"] - a["lat"])
    lon = a["lon"] + t * (b["lon"] - a["lon"])
    return lat, lon


# =========================
# STREET VIEW
# =========================
def get_metadata(lat, lon):
    """
    Check if Street View imagery exists near this location.
    """
    params = {
        "location": f"{lat},{lon}",
        "source": "outdoor",
        "radius": 20,
        "key": API_KEY,
    }
    response = session.get(META_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def get_image_bytes(lat, lon, heading):
    """
    Download one Street View image into memory.
    """
    params = {
        "size": "800x600",
        "location": f"{lat},{lon}",
        "heading": heading,
        "pitch": 0,
        "fov": 80,
        "radius": 20,
        "source": "outdoor",
        "return_error_code": "true",
        "key": API_KEY,
    }
    response = session.get(IMG_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.content


# =========================
# IMAGE SCORING
# =========================
def score_road_centered(image_bytes):
    """
    Heuristic score to prefer head-on road views.

    Higher score if:
    - strong non-horizontal line structure in lower half
    - line convergence near image center
    Lower score if:
    - upper image is very sky/water dominant
    - lower image is very sand/beige dominant
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return -1e9

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    roi = gray[h // 2 :, :]
    edges = cv2.Canny(roi, 80, 160)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=25,
    )

    score = 0.0

    if lines is not None:
        center_x = w / 2
        converging_bonus = 0.0
        line_score = 0.0

        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 30:
                continue

            angle = abs(math.degrees(math.atan2(dy, dx)))
            if angle < 15:
                continue

            if dx != 0:
                m = dy / dx
                b = y1 - m * x1
                if m != 0:
                    x_at_top = -b / m
                    dist_to_center = abs(x_at_top - center_x)
                    converging_bonus += max(0, 300 - dist_to_center) / 300.0

            line_score += length / 100.0

        score += line_score + 2.5 * converging_bonus

    upper = img[: h // 3, :, :]
    upper_hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(upper_hsv, (90, 30, 30), (140, 255, 255))
    blue_ratio = blue_mask.mean() / 255.0
    score -= 2.0 * blue_ratio

    lower = img[h // 2 :, :, :]
    lower_hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    sand_mask = cv2.inRange(lower_hsv, (10, 20, 80), (35, 140, 255))
    sand_ratio = sand_mask.mean() / 255.0
    score -= 2.0 * sand_ratio

    return float(score)


# =========================
# MAIN SAVE/UPLOAD LOGIC
# =========================
def choose_best_view(lat, lon):
    """
    Download 4 headings and keep the best-scoring image.
    """
    best = None

    for heading in HEADINGS:
        try:
            image_bytes = get_image_bytes(lat, lon, heading)
            score = score_road_centered(image_bytes)
            if best is None or score > best["score"]:
                best = {
                    "heading": heading,
                    "score": score,
                    "bytes": image_bytes,
                }
        except Exception as e:
            print(f"  Heading {heading} failed: {e}")

    return best


def main():
    if API_KEY == "YOUR_GOOGLE_STREET_VIEW_API_KEY":
        raise ValueError("Please set API_KEY in the script.")

    print("Authenticating with Google Drive...")
    drive_service = get_drive_service()
    print("Drive authentication successful.")

    saved = 0
    attempts = 0
    max_attempts = NUM_IMAGES * 30
    road_cache = []

    while saved < NUM_IMAGES and attempts < max_attempts:
        attempts += 1

        if len(road_cache) < 20:
            region = random.choice(REGIONS)
            try:
                ways = overpass_query_for_roads(region)
                random.shuffle(ways)
                road_cache.extend([(region[0], w) for w in ways if len(w.get("geometry", [])) >= 2])
                print(f"Loaded {len(ways)} roads from {region[0]}")
                time.sleep(1.0)
            except Exception as e:
                print(f"Overpass error: {e}")
                time.sleep(2.0)
                continue

        region_name, way = road_cache.pop()

        point = random_point_on_way(way)
        if point is None:
            continue

        lat, lon = point

        try:
            metadata = get_metadata(lat, lon)
        except Exception as e:
            print(f"Metadata error: {e}")
            continue

        if metadata.get("status") != "OK":
            continue

        pano_loc = metadata.get("location", {})
        pano_lat = pano_loc.get("lat", lat)
        pano_lon = pano_loc.get("lng", lon)

        best = choose_best_view(pano_lat, pano_lon)
        if best is None:
            continue

        filename = f"{saved+1:04d}_{region_name}_{pano_lat:.5f}_{pano_lon:.5f}_h{best['heading']}.jpg"

        try:
            uploaded = upload_bytes_to_drive(
                drive_service,
                best["bytes"],
                filename,
                folder_id=DRIVE_FOLDER_ID,
                mime_type="image/jpeg",
            )
            saved += 1
            print(
                f"Uploaded {saved}/{NUM_IMAGES}: {uploaded['name']} | "
                f"heading={best['heading']} | score={best['score']:.2f} | "
                f"drive_id={uploaded['id']}"
            )
            time.sleep(0.3)
        except Exception as e:
            print(f"Drive upload error: {e}")

    print(f"Finished. Uploaded {saved}/{NUM_IMAGES} images.")


if __name__ == "__main__":
    main()