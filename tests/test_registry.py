from __future__ import annotations

import pytest

from quantphemes_rl.registry import RegistryError, build, list_registered, register


def test_register_build_and_list() -> None:
    @register("test_kind", "example")
    class ExamplePlugin:
        def __init__(self, value: int) -> None:
            self.value = value

    instance = build("test_kind", "example", value=7)

    assert isinstance(instance, ExamplePlugin)
    assert instance.value == 7
    assert "example" in list_registered("test_kind")


def test_unknown_name_error() -> None:
    @register("test_unknown_name", "known")
    class KnownPlugin:
        pass

    with pytest.raises(RegistryError, match="Unknown test_unknown_name plugin 'missing'"):
        build("test_unknown_name", "missing")


def test_unknown_kind_error() -> None:
    with pytest.raises(RegistryError, match="Unknown registry kind"):
        list_registered("not_registered")
