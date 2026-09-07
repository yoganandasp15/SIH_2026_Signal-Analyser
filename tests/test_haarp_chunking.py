import os
import unittest
from dsp.loaders import load_signal_file
from dsp.adaptive_pipeline import run_adaptive_pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAARP_SAMPLE_PATH = os.path.join(PROJECT_ROOT, "verified_samples", "HAARP-1.wav")


class TestHaarpChunking(unittest.TestCase):
    """Regression test ensuring HAARP-1.wav is never misclassified across any chunk sizes."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(HAARP_SAMPLE_PATH):
            raise unittest.SkipTest(f"HAARP-1.wav not found at {HAARP_SAMPLE_PATH}")
        cls.sig, cls.fs, cls.meta = load_signal_file(HAARP_SAMPLE_PATH, max_samples=1_000_000)

    def test_haarp_across_all_chunk_sizes(self):
        chunk_sizes = [20_000, 50_000, 100_000, 250_000, 500_000, len(self.sig)]
        for N in chunk_sizes:
            chunk = self.sig[:N]
            res = run_adaptive_pipeline(chunk, self.fs, metadata=self.meta)
            det = res.get("autonomous_detection", {})
            class_id = det.get("signal_class_id", "")
            pipe = det.get("extraction_pipeline", "")
            conf = det.get("confidence", 0.0)

            self.assertEqual(
                class_id,
                "RADAR_HAARP_IONO",
                f"Failed on chunk size N={N}: expected RADAR_HAARP_IONO, got {class_id}"
            )
            self.assertEqual(pipe, "pulsed_radar", f"Expected pulsed_radar pipeline on N={N}, got {pipe}")
            self.assertGreaterEqual(conf, 0.90, f"Confidence {conf} below 0.90 on N={N}")


if __name__ == "__main__":
    unittest.main()
