from model_settings import installed_chat_models, load_chat_model, save_chat_model


class FakeModel:
    def __init__(self, name):
        self.model = name


class FakeOllama:
    def list(self):
        return {"models": [FakeModel("chat:latest"), FakeModel("embed:latest")]}

    def show(self, name):
        capabilities = ["embedding"] if name == "embed:latest" else ["completion"]
        return {"capabilities": capabilities}


def test_selected_model_is_persisted(tmp_path):
    settings_path = tmp_path / "settings.json"

    save_chat_model("qwen:test", settings_path)

    assert load_chat_model(settings_path) == "qwen:test"


def test_only_completion_models_are_listed():
    assert installed_chat_models(FakeOllama()) == ["chat:latest"]
