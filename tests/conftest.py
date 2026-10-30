"""Shared test fixtures and environment setup."""

import os

# Guarantee a valid API key exists before any module imports Settings.
# The real value doesn't matter because we mock the Groq client in every test.
os.environ.setdefault("MODEL_API_KEY", "gsk_test_key_for_unit_tests")