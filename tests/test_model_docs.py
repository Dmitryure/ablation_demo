from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.generate_model_docs import DEFAULT_DOCS_DIR, render_dot, render_markdown
from scripts.model_architecture_spec import (
    DEFAULT_CONFIG,
    architecture_spec_to_json,
    build_architecture_spec,
    load_yaml,
)


class ModelDocsTest(unittest.TestCase):
    def test_build_architecture_spec_extracts_exact_branch_stage_order(self):
        config = load_yaml(DEFAULT_CONFIG)

        spec = build_architecture_spec(config, config_path=DEFAULT_CONFIG)
        components = {component.id: component for component in spec.components}

        self.assertEqual(
            [stage.title for stage in components["eye_gaze"].stages],
            ["Input", "Project", "Temporal Pool"],
        )
        self.assertEqual(
            [stage.title for stage in components["face_mesh"].stages],
            ["Input", "Project", "Point Pool", "Flatten Frame Tokens", "Clip Pool"],
        )
        self.assertEqual(
            [stage.title for stage in components["fau"].stages],
            ["Input", "Project", "Frame Pool", "Flatten Frame Tokens", "Clip Pool"],
        )
        self.assertEqual(
            [stage.title for stage in components["depth"].stages],
            ["Input", "Project", "Temporal Pool"],
        )
        self.assertEqual(
            [stage.title for stage in components["fusion_core"].stages],
            [
                "Input",
                "Add Time Embedding",
                "Add Modality Embedding",
                "Expand CLS",
                "Prepend CLS",
                "Build Padding Mask",
                "Transformer Encoder",
                "LayerNorm",
            ],
        )

    def test_build_architecture_spec_keeps_fixed_bank_for_disabled_modalities(self):
        config = load_yaml(DEFAULT_CONFIG)
        config["modalities"] = ["face_mesh"]

        spec = build_architecture_spec(config, config_path=DEFAULT_CONFIG)
        components = {component.id: component for component in spec.components}
        modality_components = [
            component for component in spec.components if component.kind == "modality"
        ]
        expected_total_tokens = sum(component.token_count or 0 for component in modality_components)
        expected_enabled_tokens = components["face_mesh"].token_count

        self.assertEqual(spec.total_tokens, expected_total_tokens)
        self.assertEqual(spec.enabled_token_count, expected_enabled_tokens)
        self.assertEqual(spec.frames["rgb"], 16)
        self.assertEqual(spec.frames["rppg"], 32)
        self.assertFalse(components["rgb"].enabled)
        self.assertFalse(components["fau"].enabled)
        self.assertFalse(components["rppg"].enabled)
        self.assertFalse(components["eye_gaze"].enabled)
        self.assertTrue(components["face_mesh"].enabled)
        self.assertFalse(components["depth"].enabled)
        self.assertEqual(
            components["token_bank"].token_formula,
            f"enabled_slots={expected_enabled_tokens}, fixed_slots={expected_total_tokens}",
        )

    def test_generated_json_markdown_and_dot_include_live_code_details(self):
        config = load_yaml(DEFAULT_CONFIG)
        spec = build_architecture_spec(config, config_path=DEFAULT_CONFIG)

        json_text = architecture_spec_to_json(spec)
        markdown = render_markdown(spec)
        dot_source = render_dot(spec, docs_dir=DEFAULT_DOCS_DIR)

        self.assertIn('"title": "Eye Gaze Branch"', json_text)
        self.assertIn('"title": "Depth Branch"', json_text)
        self.assertIn('"title": "Add Time Embedding"', json_text)
        self.assertIn("Project: MLP(8->128->128)", markdown)
        self.assertIn("Point Pool: LatentQueryPooling(output_tokens=1)", markdown)
        self.assertIn(
            "Transformer Encoder: TransformerEncoderLayer x2 (heads=4, hidden=512)", markdown
        )
        self.assertIn("Frames: `rgb=16", markdown)
        self.assertIn("model_architecture.json", markdown)
        self.assertIn('href="../branches/eye_gaze.py"', dot_source)
        self.assertIn("MLP(8-&gt;128-&gt;128)", dot_source)
        self.assertIn("Add Time Embedding", dot_source)
        self.assertIn("Token Bank", dot_source)
        self.assertIn("Depth Branch", dot_source)

    def test_dot_smoke_render_succeeds_when_graphviz_is_available(self):
        if shutil.which("dot") is None:
            self.skipTest("Graphviz dot not installed")

        config = load_yaml(DEFAULT_CONFIG)
        spec = build_architecture_spec(config, config_path=DEFAULT_CONFIG)
        dot_source = render_dot(spec, docs_dir=DEFAULT_DOCS_DIR)

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            dot_path = temp_dir / "model_architecture.dot"
            svg_path = temp_dir / "model_architecture.svg"
            dot_path.write_text(dot_source, encoding="utf-8")
            subprocess.run(
                ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
                check=True,
            )

            self.assertTrue(svg_path.exists())
            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertIn("Eye Gaze Branch", svg_text)
            self.assertIn("TokenBankFusion", svg_text)


if __name__ == "__main__":
    unittest.main()
