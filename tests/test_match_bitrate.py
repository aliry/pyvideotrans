"""The final re-encode keeps the original file's video and audio bitrates."""
import json
from types import SimpleNamespace

import pytest

from videotrans.configure.config import app_cfg, settings
from videotrans.task import _stage_assemble
from videotrans.task._stage_assemble import AssembleMixin
from videotrans.util import _ffprobe


def _probe(video_stream_extra, audio_stream_extra, format_extra):
    return json.dumps({
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920,
             "pix_fmt": "yuv420p", "r_frame_rate": "30/1", "avg_frame_rate": "30/1",
             "duration": "283.433333", **video_stream_extra},
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "sample_rate": "44100",
             "channels": 2, "duration": "283.477333", **audio_stream_extra},
        ],
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "283.477333", **format_extra},
    })


@pytest.mark.parametrize("probe_json, want_video, want_audio", [
    # mp4: per-stream bit_rate
    (_probe({"bit_rate": "3903798"}, {"bit_rate": "128000"}, {"bit_rate": "4039751"}), 3903798, 128000),
    # mkv/webm: no stream bit_rate, only BPS statistics tags
    (_probe({"tags": {"BPS": "2500000"}}, {"tags": {"BPS-eng": "160000"}}, {"bit_rate": "2700000"}), 2500000, 160000),
    # video bit_rate missing: container total minus audio
    (_probe({}, {"bit_rate": "128000"}, {"bit_rate": "1000000"}), 872000, 128000),
    # nothing known
    (_probe({}, {}, {}), 0, 0),
])
def test_video_info_reports_original_bitrates(tmp_path, monkeypatch, probe_json, want_video, want_audio):
    media = tmp_path / "in.mp4"
    media.write_bytes(b"")
    monkeypatch.setattr(_ffprobe, "runffprobe", lambda cmd: probe_json)

    info = _ffprobe.get_video_info(str(media))

    assert (info["video_bitrate"], info["audio_bitrate"]) == (want_video, want_audio)


class _Assembler(AssembleMixin):
    def __init__(self, video_bitrate, audio_bitrate):
        self.video_codec_num = 264
        self.video_info = {"video_bitrate": video_bitrate, "audio_bitrate": audio_bitrate,
                           "streams_audio": 1, "time": 1000}
        self.cfg = SimpleNamespace(name="/videos/original.mp4")


@pytest.fixture
def default_quality(monkeypatch):
    monkeypatch.setattr(settings, "crf", 23, raising=False)
    monkeypatch.setattr(settings, "preset", "medium", raising=False)


def test_software_encode_targets_original_video_bitrate(default_quality):
    assert _Assembler(3903798, 128000)._soft_enc_args() == ["-b:v", "3903798", "-preset", "medium"]


def test_software_encode_falls_back_to_crf_when_bitrate_unknown(default_quality):
    assert _Assembler(0, 0)._soft_enc_args() == ["-crf", "23", "-preset", "medium"]


def test_videotoolbox_encode_targets_original_video_bitrate(default_quality, monkeypatch):
    monkeypatch.setattr(app_cfg, "video_codec", "h264_videotoolbox", raising=False)

    _, _, vcodec, enc_args = _Assembler(288654, 64000)._get_hard_cfg("end.ass")

    assert (vcodec, enc_args) == ("h264_videotoolbox", ["-b:v", "288654"])


def test_videotoolbox_falls_back_to_quality_when_bitrate_unknown(default_quality, monkeypatch):
    monkeypatch.setattr(app_cfg, "video_codec", "h264_videotoolbox", raising=False)

    _, _, _, enc_args = _Assembler(0, 0)._get_hard_cfg("end.ass")

    assert enc_args == ["-q:v", "67"]


def test_unrecognised_hw_codec_falls_back_to_software_at_original_bitrate(default_quality, monkeypatch):
    monkeypatch.setattr(app_cfg, "video_codec", "libx264", raising=False)

    _, _, vcodec, enc_args = _Assembler(3903798, 128000)._get_hard_cfg("end.ass")

    assert (vcodec, enc_args) == ("libx264", ["-b:v", "3903798", "-preset", "medium"])


@pytest.mark.parametrize("audio_bitrate, want", [(64000, "64000"), (0, "128k")])
def test_original_audio_is_reencoded_at_its_own_bitrate(monkeypatch, audio_bitrate, want):
    calls = []
    monkeypatch.setattr(_stage_assemble, "runffmpeg", lambda cmd, **kw: calls.append(cmd))

    _Assembler(288654, audio_bitrate)._get_origin_audio("will_embed.m4a", duration_ms=1000)

    cmd = calls[0]
    assert cmd[cmd.index("-b:a") + 1] == want


def test_software_fallback_keeps_normalised_preset(default_quality, monkeypatch):
    monkeypatch.setattr(app_cfg, "video_codec", "libx264", raising=False)
    monkeypatch.setattr(settings, "preset", "veryslow", raising=False)

    _, _, _, enc_args = _Assembler(3903798, 128000)._get_hard_cfg("end.ass")

    assert enc_args == ["-b:v", "3903798", "-preset", "slow"]
