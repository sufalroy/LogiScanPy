import os

import requests


def create_directories():
    if not os.path.exists("weights"):
        os.makedirs("weights")
    if not os.path.exists("videos"):
        os.makedirs("videos")
    if not os.path.exists("outputs"):
        os.makedirs("outputs")


def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    create_directories()

    file_id = "1oRj2OqCxwrgqs5o0VCm5a4sbbyDWx497"
    destination = os.path.join("weights", "sack_yolov8_50e_v2.pt")
    download_file_from_google_drive(file_id, destination)