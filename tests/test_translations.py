import ast
from pathlib import Path
from string import Formatter

import pytest

from image_dataset_studio.core import compile_filter
from image_dataset_studio.translations import msg, set_language


def test_catalog_has_both_languages_and_matching_format_fields():
    for entry in vars(msg).values():
        assert set(entry) == {'JP', 'EN'}
        fields = [{name for _, name, _, _ in Formatter().parse(entry[language]) if name}
                  for language in ('JP', 'EN')]
        assert fields[0] == fields[1]


def test_user_facing_japanese_text_is_only_in_translation_catalog():
    source = Path(__file__).resolve().parents[1] / 'src' / 'image_dataset_studio'
    for path in source.rglob('*.py'):
        if path.name == 'translations.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        literals = [node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        assert not [value for value in literals
                    if any('\u3040' <= char <= '\u9fff' for char in value)], path


def test_core_error_uses_current_language():
    try:
        set_language('EN')
        with pytest.raises(ValueError, match='Unsupported search term'):
            compile_filter('bad:foo')
    finally:
        set_language('JP')
