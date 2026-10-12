import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.gpu_status import fetch_report, gather_reports


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(["ssh"], returncode, stdout, stderr)


class TestFetchReport(unittest.TestCase):
    def test_returns_server_output(self):
        runner = MagicMock(return_value=completed(stdout="🖥️ [hufs5090] report\n"))
        self.assertEqual(fetch_report("gpu-status-hufs5090", runner=runner), "🖥️ [hufs5090] report")
        args = runner.call_args[0][0]
        self.assertEqual(args[0], "ssh")
        self.assertIn("BatchMode=yes", args)
        self.assertIn("gpu-status-hufs5090", args)

    def test_retries_once_after_connection_reset(self):
        runner = MagicMock(side_effect=[
            completed(255, stderr="kex_exchange_identification: Connection reset by peer"),
            completed(stdout="🖥️ [hufs5090] report"),
        ])
        self.assertEqual(fetch_report("gpu-status-hufs5090", runner=runner), "🖥️ [hufs5090] report")
        self.assertEqual(runner.call_count, 2)

    def test_warns_in_place_when_all_attempts_fail(self):
        runner = MagicMock(return_value=completed(255, stderr="Permission denied (publickey)"))
        out = fetch_report("gpu-status-hufs4090", runner=runner)
        self.assertIn("⚠️ [gpu-status-hufs4090]", out)
        self.assertIn("Permission denied", out)
        self.assertEqual(runner.call_count, 2)

    def test_timeout_is_a_failure_not_a_crash(self):
        runner = MagicMock(side_effect=subprocess.TimeoutExpired(["ssh"], 40))
        self.assertIn("⚠️", fetch_report("gpu-status-hufs4090", runner=runner))

    def test_empty_output_is_a_failure(self):
        runner = MagicMock(return_value=completed(0, stdout=""))
        self.assertIn("⚠️", fetch_report("gpu-status-hufs4090", runner=runner))


class TestGatherReports(unittest.TestCase):
    def test_keeps_configured_order(self):
        out = gather_reports(["b", "a"], fetch=lambda host: f"R-{host}")
        self.assertEqual(out, "R-b\nR-a")

    def test_no_hosts(self):
        self.assertIn("gpu_status_hosts", gather_reports([]))


if __name__ == "__main__":
    unittest.main()
