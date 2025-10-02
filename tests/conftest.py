import pytest
from fastapi.testclient import TestClient
from src.service import app
@pytest.fixture(scope="session")
def client():
    return TestClient(app)
