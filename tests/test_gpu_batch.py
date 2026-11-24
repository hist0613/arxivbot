"""GpuBatch: 서버 대신 FakeRemote 를 끼워 보내기·받기·재전송 규칙을 확인한다."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.cache import CacheManager  # noqa: E402
from api.gpu_batch import GpuBatch, to_cached_summary  # noqa: E402
from api.store import Store  # noqa: E402
from prompts import is_current_summary_schema  # noqa: E402

URL = "https://arxiv.org/abs/2611.00001"
INFO = f"Paper One ({URL})"
URL2 = "https://arxiv.org/abs/2611.00002"
INFO2 = f"Paper Two ({URL2})"
SUMMARY = {"prior_approaches": "기존", "core_contribution": "기여",
           "technical_challenges": "난제", "empirical_impact": "실험"}


class FakeRemote:
    """서버 폴더를 로컬 임시 폴더로 흉내 낸다. status 는 테스트가 직접 써 넣는다."""

    def __init__(self, root: Path):
        self.root = root
        self.commands = []

    def run(self, command, timeout=120):
        self.commands.append(command)
        if command.startswith("cat batches/"):
            path = self.root / command.split()[1]
            return path.read_text() if path.exists() else "missing\n"
        if command.startswith("test -d"):
            return "yes\n" if (self.root / command.split()[2]).is_dir() else "no\n"
        if command.startswith("mkdir -p"):
            (self.root / command.split()[2]).mkdir(parents=True, exist_ok=True)
        return ""

    def put(self, local, remote_rel, timeout=300):
        dst = self.root / remote_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dst)

    def get(self, remote_rel, local, recursive=False, timeout=600):
        src = self.root / remote_rel
        if recursive:
            shutil.copytree(src, Path(local) / src.name, dirs_exist_ok=True)
        else:
            shutil.copyfile(src, local)


class TestGpuBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = Store(str(self.tmp / "c.db"))
        self.cache = CacheManager(self.store)
        self.remote = FakeRemote(self.tmp / "server")
        self.gpu = GpuBatch(self.cache, self.remote, str(self.tmp / "state"), str(self.tmp / "figs"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp)

    def _finish(self, batch_id, records, status="done", fig_for=None):
        b = self.remote.root / "batches" / batch_id
        (b / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        if fig_for:
            (b / "figs").mkdir(exist_ok=True)
            (b / "figs" / f"{fig_for}.png").write_bytes(b"png")
        (b / "status").write_text(status + "\n")

    def test_cached_summary_matches_batch_schema(self):
        self.assertTrue(is_current_summary_schema(to_cached_summary(SUMMARY)))

    def test_submit_skips_summarized_and_pending(self):
        self.cache.update_paper_summarizations(INFO2, to_cached_summary(SUMMARY))
        batch_id = self.gpu.submit([(URL, "Paper One", INFO), (URL2, "Paper Two", INFO2)])
        request = (self.remote.root / "batches" / batch_id / "request.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(r)["arxiv_id"] for r in request], ["2611.00001"])
        self.assertTrue(any("run_batch.sh" in c for c in self.remote.commands))
        # 결과가 오기 전에는 같은 논문을 또 보내지 않는다
        self.assertIsNone(self.gpu.submit([(URL, "Paper One", INFO)]))

    def test_collect_imports_summary_and_figure(self):
        batch_id = self.gpu.submit([(URL, "Paper One", INFO)])
        self._finish(batch_id, [{"arxiv_id": "2611.00001", "paper_info": INFO, "model": "m", "summary": SUMMARY,
                                 "figure": {"figure_id": "S1.F1", "label": "Figure 1", "description": "구조도"}}],
                     fig_for="2611.00001")
        self.assertEqual(self.gpu.collect(), 1)
        self.assertTrue(is_current_summary_schema(self.cache.paper_summarizations[INFO]))
        fig = self.store.get_figure(INFO)
        self.assertEqual((fig["label"], fig["description"]), ("Figure 1", "구조도"))
        self.assertTrue(os.path.exists(fig["path"]))
        self.assertEqual(self.gpu.pending_infos(), set())

    def test_running_batch_stays_pending(self):
        batch_id = self.gpu.submit([(URL, "Paper One", INFO)])
        (self.remote.root / "batches" / batch_id / "status").write_text("running 12\n")
        self.assertEqual(self.gpu.collect(), 0)
        self.assertEqual(self.gpu.pending_infos(), {INFO})

    def test_failed_paper_is_resent_at_most_twice(self):
        for _ in range(2):
            batch_id = self.gpu.submit([(URL, "Paper One", INFO)])
            self.assertIsNotNone(batch_id)
            self._finish(batch_id, [{"arxiv_id": "2611.00001", "paper_info": INFO, "error": "x"}])
            self.gpu.collect()
        self.assertEqual(self.gpu.failed_attempts(), {INFO: 2})
        self.assertIsNone(self.gpu.submit([(URL, "Paper One", INFO)]))


if __name__ == "__main__":
    unittest.main()
