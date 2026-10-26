"""
Domain exceptions raised by library modules.

These replace ``SystemExit`` in non-CLI code so that callers (tests, future library consumers) can
handle failures without catching ``SystemExit``. The CLI commands translate them into exit codes
at their boundary.
"""


class PhotoTaggerError(Exception):
    """Base for all photo-tagger domain errors."""


class ProviderError(PhotoTaggerError):
    """A provider validation or connectivity check failed."""


class DiscoveryError(PhotoTaggerError):
    """File discovery or skip-list loading failed."""


class BatchError(PhotoTaggerError):
    """One or more photos in a batch failed to process."""


class ObservabilityError(PhotoTaggerError):
    """``--otel`` was asked for but tracing cannot be set up (no endpoint, or no OTel SDK)."""


class FieldError(PhotoTaggerError):
    """A copyright template or a ``--set TAG=VALUE`` assignment cannot be used as written."""


class ConfigFileError(PhotoTaggerError):
    """
    The config file on disk cannot be updated in place.

    Reading a config is always best-effort (a broken file is logged and ignored, see
    :func:`photo_tagger.config_file.load_config`), but *writing* one back is not: the GUI's Save
    Settings has to preserve everything it does not manage, and it cannot do that with a file it
    could not parse. Raised so the window reports the problem and leaves the file alone.
    """


class DngDecodeError(PhotoTaggerError):
    """The Deflate DNG fallback cannot decode a file: another kind of DNG, or a damaged one."""
