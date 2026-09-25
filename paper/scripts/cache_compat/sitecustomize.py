"""Migrate legacy Inspect 1W cache keys to unlimited keys on access."""

from __future__ import annotations

import os
import pickle
from copy import copy

import inspect_ai.model._cache as cache
from inspect_ai.model._model_output import ModelOutput


_original_cache_fetch = cache.cache_fetch
_original_cache_key = cache._cache_key


def _legacy_key(entry: cache.CacheEntry) -> str:
    legacy_entry = copy(entry)
    legacy_entry.policy = entry.policy.model_copy(update={"expiry": "1W"})
    return _original_cache_key(legacy_entry)


def _migrating_cache_fetch(entry: cache.CacheEntry) -> ModelOutput | None:
    output = _original_cache_fetch(entry)
    if output is not None or entry.policy.expiry is not None:
        return output

    unlimited_path = cache.cache_path(model=entry.model) / entry.key
    legacy_path = cache.cache_path(model=entry.model) / _legacy_key(entry)
    try:
        with legacy_path.open("rb") as source:
            _expiry, output = pickle.load(source)
        if not isinstance(output, ModelOutput):
            return None
        unlimited_path.parent.mkdir(parents=True, exist_ok=True)
        with legacy_path.open("wb") as destination:
            pickle.dump((None, output), destination)
        os.replace(legacy_path, unlimited_path)
        return output
    except (OSError, pickle.PickleError):
        return None


cache.cache_fetch = _migrating_cache_fetch
