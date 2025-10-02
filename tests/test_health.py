def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"

def test_warmup(client):
    r = client.post("/warmup")
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body
