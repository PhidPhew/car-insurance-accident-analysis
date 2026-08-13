import json

from config.settings import settings
from rri.utils.logging import configure_logging, get_logger, new_run_id


def test_new_run_id_generates_unique_hex_ids() -> None:
    first = new_run_id()
    second = new_run_id()

    assert first != second
    assert len(first) == 32  # uuid4().hex


def test_configure_logging_writes_jsonl_with_run_id() -> None:
    run_id = configure_logging()
    log = get_logger("test")
    log.info("unit_test_event", detail="value")

    log_file = settings.logs_dir / f"{run_id}.jsonl"
    assert log_file.exists()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    last_entry = json.loads(lines[-1])

    assert last_entry["run_id"] == run_id
    assert last_entry["event"] == "unit_test_event"
    assert last_entry["detail"] == "value"

    log_file.unlink()
