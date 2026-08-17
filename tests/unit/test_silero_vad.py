import numpy as np

from companion.infrastructure.models.vad.silero import SileroVADProvider


class _SpeechModel:
    def run(self, _outputs, inputs):
        return [np.array(0.9, dtype=np.float32), inputs["state"]]


def test_silero_buffers_20ms_microphone_chunks_into_full_frames():
    provider = SileroVADProvider.__new__(SileroVADProvider)
    provider._params = {"threshold": 0.5}
    provider._model = _SpeechModel()
    provider._state = np.zeros((2, 1, 128), dtype=np.float32)
    provider._pending = np.empty(0, dtype=np.float32)
    provider._last_speech = False
    provider.sr = 16000

    # 20 ms at 16 kHz is 320 samples: shorter than Silero's 512-sample frame.
    assert not provider._infer(np.ones(320, dtype=np.float32))
    assert provider._infer(np.ones(320, dtype=np.float32))
