"""
Domain exceptions raised by library modules.

These replace ``SystemExit`` in non-CLI code so that callers (tests, future
library consumers) can handle failures without catching ``SystemExit``.
The CLI boundary in ``main.py`` translates them into exit codes.
"""


class PhotoTaggerError(Exception):
    """Base for all photo-tagger domain errors."""


class ProviderError(PhotoTaggerError):
    """A provider validation or connectivity check failed."""


class DiscoveryError(PhotoTaggerError):
    """File discovery or skip-list loading failed."""


class BatchError(PhotoTaggerError):
    """One or more photos in a batch failed to process."""
