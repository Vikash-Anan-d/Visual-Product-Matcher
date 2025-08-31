import io
import json
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    """Test that the index page loads successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Visual Product Matcher' in response.data

def test_search_no_file_or_url(client):
    """Test /search endpoint with no file or URL provided."""
    response = client.post('/search', data={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data

def test_search_with_invalid_file(client):
    """Test /search endpoint with invalid file upload."""
    data = {
        'file': (io.BytesIO(b"not an image"), 'test.txt')
    }
    response = client.post('/search', data=data, content_type='multipart/form-data')
    # Should return 500 or error due to invalid image
    assert response.status_code in (400, 500)
    data = response.get_json()
    assert 'error' in data

def test_search_with_valid_image(client):
    """Test /search endpoint with a valid image file."""
    # Use a small valid image from the Data/archive/test/test folder
    image_path = 'Data/archive/test/test/0004b03ad7eabfb3989727c461310a84.jpg'
    with open(image_path, 'rb') as img_file:
        data = {
            'file': (img_file, '0004b03ad7eabfb3989727c461310a84.jpg')
        }
        response = client.post('/search', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        results = response.get_json()
        assert isinstance(results, list)
        if results:
            first = results[0]
            assert 'name' in first
            assert 'image_path' in first
            assert 'similarity' in first
