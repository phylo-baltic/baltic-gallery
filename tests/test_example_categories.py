from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_examples import EXAMPLES_SRC, group_items_by_category
from content_metadata import landing_categories, load_metadata


def example(name: str, tags: list[str], rst_doc: str | None = None):
    return SimpleNamespace(
        example_name=name,
        rst_doc=rst_doc or name,
        source_path=EXAMPLES_SRC / "test" / f"{name}.py",
        tags=tags,
    )


class LandingCategoryMetadataTests(unittest.TestCase):
    def test_categories_are_canonicalised_and_keep_their_order(self):
        section = {
            "tag_labels": {"basic": "Basic", "colour": "Colour"},
            "landing_categories": [
                {"title": "First", "tags": ["basic"]},
                {"title": "Second", "tags": ["colour"]},
            ],
        }

        categories = landing_categories(section)

        self.assertEqual([category["title"] for category in categories], ["First", "Second"])
        self.assertEqual(categories[0]["tags"], ["Basic"])

    def test_unknown_category_tag_is_rejected(self):
        section = {
            "tag_labels": {"basic": "Basic"},
            "landing_categories": [{"title": "Other", "tags": ["missing"]}],
        }

        with self.assertRaisesRegex(SystemExit, "Unknown landing category tag"):
            landing_categories(section)

    def test_duplicate_category_slug_is_rejected(self):
        section = {
            "tag_labels": {"basic": "Basic"},
            "landing_categories": [
                {"title": "Tree layout", "tags": ["basic"]},
                {"title": "Tree-layout", "tags": ["basic"]},
            ],
        }

        with self.assertRaisesRegex(SystemExit, "Duplicate landing category slug"):
            landing_categories(section)


class ExampleGroupingTests(unittest.TestCase):
    def test_repository_categories_cover_every_example(self):
        metadata = load_metadata("examples")
        items = [
            example(Path(source_key).stem, tags, source_key)
            for source_key, tags in metadata["items"].items()
        ]

        sections = group_items_by_category(items, landing_categories(metadata))
        covered = {item.rst_doc for _, matches in sections for item in matches}

        self.assertEqual(covered, {item.rst_doc for item in items})

    def test_example_can_appear_in_every_matching_category(self):
        shared = example("shared", ["basic", "colour"])
        basic_only = example("basic-only", ["basic"])
        categories = [
            {"title": "Basic", "slug": "basic", "tags": ["basic"]},
            {"title": "Colour", "slug": "colour", "tags": ["colour"]},
        ]

        sections = group_items_by_category([shared, basic_only], categories)

        self.assertEqual([item.rst_doc for item in sections[0][1]], ["basic-only", "shared"])
        self.assertEqual([item.rst_doc for item in sections[1][1]], ["shared"])

    def test_matching_multiple_tags_does_not_duplicate_a_card(self):
        shared = example("shared", ["basic", "colour"])
        categories = [
            {
                "title": "Rendering",
                "slug": "rendering",
                "tags": ["basic", "colour"],
            }
        ]

        sections = group_items_by_category([shared], categories)

        self.assertEqual([item.rst_doc for item in sections[0][1]], ["shared"])

    def test_uncovered_example_is_rejected(self):
        uncovered = example("uncovered", ["other"])
        categories = [
            {"title": "Basic", "slug": "basic", "tags": ["basic"]}
        ]

        with self.assertRaisesRegex(SystemExit, "Examples missing from all landing categories"):
            group_items_by_category([example("basic", ["basic"]), uncovered], categories)


if __name__ == "__main__":
    unittest.main()
