import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSchema(unittest.TestCase):
    def test_summarization_response_has_four_sections(self):
        from prompts import SummarizationResponse
        self.assertEqual(
            set(SummarizationResponse.model_fields),
            {"prior_approaches", "core_contribution",
             "technical_challenges", "empirical_impact"},
        )


class TestEncoder(unittest.TestCase):
    def test_fallback_for_unmapped_model(self):
        from api.agent import Encoder
        # gpt-5.4-nano는 tiktoken 매핑이 없어 KeyError → o200k_base로 fallback
        enc = Encoder("gpt-5.4-nano")
        self.assertEqual(enc.encoding.name, "o200k_base")

    def test_known_model_still_works(self):
        from api.agent import Encoder
        self.assertEqual(Encoder("gpt-4o-mini").encoding.name, "o200k_base")


if __name__ == "__main__":
    unittest.main()
