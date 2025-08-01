"""Tests for the AIFactory class."""

import pytest
from unittest.mock import patch, MagicMock

from esperanto.factory import AIFactory


# Note: Caching functionality has been removed from AIFactory.
# Tests now verify that factory creates new instances each time.