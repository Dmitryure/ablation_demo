from __future__ import annotations

import unittest

from scripts.generate_model_docs import DEFAULT_CONFIG, build_modality_docs, load_yaml, render_dot, render_markdown


class ModelDocsTest(unittest.TestCase):
    def test_generated_markdown_lists_exact_branch_stages(self):
        config = load_yaml(DEFAULT_CONFIG)
        docs, summary = build_modality_docs(config)

        markdown = render_markdown(DEFAULT_CONFIG, docs, summary)

        self.assertIn("Project: MLP(8->128->128)", markdown)
        self.assertIn("Project: MLP(3->128->128)", markdown)
        self.assertIn("Frame Pool: LatentQueryPooling(output_tokens=2)", markdown)
        self.assertIn("Clip Pool: TemporalLatentQueryPooling(output_tokens=32, positional_bias=sinusoidal)", markdown)

    def test_generated_dot_keeps_token_bank_and_fusion_cards(self):
        config = load_yaml(DEFAULT_CONFIG)
        docs, summary = build_modality_docs(config)

        dot_source = render_dot(docs, summary)

        self.assertIn("Eye Gaze Branch", dot_source)
        self.assertIn("Face Mesh Branch", dot_source)
        self.assertIn("MLP(8-&gt;128-&gt;128)", dot_source)
        self.assertIn("MLP(3-&gt;128-&gt;128)", dot_source)
        self.assertIn("Token Bank", dot_source)
        self.assertIn("TokenBankFusion", dot_source)


if __name__ == "__main__":
    unittest.main()
