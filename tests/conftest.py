import pytest

from helpdesk.app import HelpdeskApp
from helpdesk.fake_model import FakePolicyModel


@pytest.fixture
def storage(tmp_path):
    return tmp_path / "storage"


@pytest.fixture
def tickets(tmp_path):
    return tmp_path / "tickets"


@pytest.fixture
def make_app(storage, tickets):
    opened = []

    def factory(profile="normal", model=None):
        app = HelpdeskApp(
            model or FakePolicyModel(),
            storage_dir=storage,
            ticket_dir=tickets,
            status_profile=profile,
        )
        opened.append(app)
        return app

    yield factory
    for app in opened:
        try:
            app.close()
        except Exception:
            pass
