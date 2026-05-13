from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

_REGISTRIES: dict[str, dict[str, type]] = {}

T = TypeVar("T", bound=type)


class RegistryError(Exception):
    """Raised when a plugin kind or name is not registered."""


def register(kind: str, name: str) -> Callable[[T], T]:
    """Register a plugin class under a kind/name pair."""

    def decorator(cls: T) -> T:
        _REGISTRIES.setdefault(kind, {})[name] = cls
        return cls

    return decorator


def build(kind: str, name: str, **kwargs: object) -> object:
    """Instantiate a registered plugin."""
    if kind not in _REGISTRIES:
        known = ", ".join(sorted(_REGISTRIES)) or "none"
        raise RegistryError(f"Unknown registry kind '{kind}'. Known kinds: {known}.")

    registry = _REGISTRIES[kind]
    if name not in registry:
        known = ", ".join(sorted(registry)) or "none"
        raise RegistryError(f"Unknown {kind} plugin '{name}'. Known {kind} plugins: {known}.")

    return registry[name](**kwargs)


def list_registered(kind: str) -> list[str]:
    """Return registered plugin names for a kind."""
    if kind not in _REGISTRIES:
        known = ", ".join(sorted(_REGISTRIES)) or "none"
        raise RegistryError(f"Unknown registry kind '{kind}'. Known kinds: {known}.")
    return sorted(_REGISTRIES[kind])
