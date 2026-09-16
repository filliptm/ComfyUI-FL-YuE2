from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from fl_yue2.yue2.training import captioning
from fl_yue2.yue2.training.data import read_json


def request_for(root, concurrency):
    for i in range(3):
        sf.write(root / f"song{i}.wav", np.zeros(100), 8000)
    return dict(directory=str(root), task="both", replace_existing=False,
                model="test", concurrent_requests=concurrency, output=str(root / "manifest.json"))


def response(style="piano"):
    return dict(style=style, lyrics="", instrumental=True, uncertainty="", complete=True)


def test_parallel_completion_keeps_song_order(tmp_path, monkeypatch):
    release_first, lock = Event(), Lock()
    active = peak = 0
    first_pair = Barrier(2)

    def listen(client, audio, request, prompt, emit, cancelled):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if audio.stem in {"song0", "song1"}:
                first_pair.wait(timeout=5)
            if audio.stem == "song0":
                assert release_first.wait(5), "Later songs did not run concurrently"
            if audio.stem == "song2":
                release_first.set()
            return response(audio.stem)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(captioning, "listen", listen)
    events = []
    request = request_for(tmp_path, 2)
    captioning.caption(request, events.append, lambda: None, object())
    assert peak == 2
    assert [song["name"] for song in read_json(request["output"])["songs"]] == ["song0", "song1", "song2"]
    for index in range(3):
        assert (tmp_path / f"song{index}.caption.txt").read_text() == f"song{index}"
    assert [event["step"] for event in events if event["type"] == "progress"] == [1, 2, 3]


def test_cancellation_stops_scheduling(tmp_path, monkeypatch):
    started, cancel, lock = Event(), Event(), Lock()
    calls = []

    def check_cancelled():
        if cancel.is_set():
            raise InterruptedError("user cancelled")

    def listen(client, audio, request, prompt, emit, cancelled):
        with lock:
            calls.append(audio.stem)
            if len(calls) == 2:
                started.set()
        assert cancel.wait(5)
        cancelled()
        pytest.fail("Cancelled request continued")

    monkeypatch.setattr(captioning, "listen", listen)
    with ThreadPoolExecutor(max_workers=1) as runner:
        future = runner.submit(captioning.caption, request_for(tmp_path, 2), lambda _: None, check_cancelled, object())
        try:
            assert started.wait(5)
        finally:
            cancel.set()
        with pytest.raises(InterruptedError, match="user cancelled"):
            future.result(timeout=5)
    assert sorted(calls) == ["song0", "song1"]
    assert not list(tmp_path.glob("*.caption.txt"))


def test_failure_keeps_completed_manifest(tmp_path, monkeypatch):
    calls = []

    def listen(client, audio, request, prompt, emit, cancelled):
        calls.append(audio.stem)
        if audio.stem == "song1":
            raise RuntimeError("quota exhausted")
        return response()

    monkeypatch.setattr(captioning, "listen", listen)
    request = request_for(tmp_path, 1)
    with pytest.raises(RuntimeError, match="quota exhausted"):
        captioning.caption(request, lambda _: None, lambda: None, object())
    assert calls == ["song0", "song1"]
    assert [song["name"] for song in read_json(request["output"])["songs"]] == ["song0"]


@pytest.mark.parametrize("value", [0, 9, True, 1.5])
def test_invalid_concurrency(value):
    with pytest.raises(ValueError, match="concurrent_requests"):
        captioning.caption({"concurrent_requests": value}, lambda _: None, lambda: None, object())


def test_listen_uses_current_interactions_response_format(tmp_path):
    sf.write(tmp_path / "song.wav", np.zeros(100), 8000)
    captured = {}
    result = dict(style="piano", lyrics="", instrumental=True, uncertainty="", complete=True)

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_text=json.dumps(result))

    client = SimpleNamespace(
        files=SimpleNamespace(
            upload=lambda file: SimpleNamespace(
                name="upload", state="ACTIVE", uri="https://example.invalid/audio", mime_type="audio/wav"
            ),
            delete=lambda name: None,
        ),
        interactions=SimpleNamespace(create=create),
    )
    captioning.listen(client, tmp_path / "song.wav", {"model": "gemini-3.8-flash"}, "prompt", lambda _: None, lambda: None)
    assert captured["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": captioning.SCHEMA,
    }
