import requests
import io
from PIL import Image


BASE_URL = "http://localhost:8000"


def test_api_activity():
    response = requests.get(f"{BASE_URL}/active")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["model_loaded"] == True


def test_api_prediction():
    img = Image.new('RGB', (150, 150), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    files = {"file": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")}
    response = requests.post(f"{BASE_URL}/predict/", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "class" in data
    assert data["class"] in ["Cat", "Dog"]
    assert 0 <= data["confidence"] <= 1