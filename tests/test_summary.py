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


if __name__ == "__main__":
    unittest.main()
