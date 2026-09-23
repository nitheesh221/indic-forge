from fastapi.testclient import TestClient

from indictrans2_finetune.app import create_app

client = TestClient(create_app())


def test_home_renders():
    response = client.get("/")
    assert response.status_code == 200
    assert "IndicTrans2 LoRA fine-tune" in response.text
    assert "eng_Latn" in response.text


def test_recipe_english_hindi():
    response = client.get(
        "/api/recipe",
        params={"src": "en", "tgt": "hi", "size": "distilled", "steps": 200},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "indictrans2-en-indic-dist-200M" in payload["model"]
    assert "--src-lang eng_Latn" in payload["command"]
    assert "--tgt-lang hin_Deva" in payload["command"]
