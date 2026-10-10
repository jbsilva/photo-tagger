"""The ``vocabulary`` command: build a keyword list from the keywords photos carry."""

from pathlib import Path  # noqa: TC003 - cyclopts resolves the flag annotations at runtime.
from typing import Annotated

from cyclopts import Parameter, validators
from loguru import logger
from rich.console import Console

from photo_tagger import telemetry
from photo_tagger.cli_options import (
    DEFAULT_EXTENSIONS,
    DEFAULT_PROVIDER_OPTIONS,
    DEFAULT_RECURSIVE,
    ProviderConfig,
    VocabularyBuildConfig,
    to_trim_rules,
)
from photo_tagger.config_file import export_exiftool_path
from photo_tagger.discovery import resolve_image_batch
from photo_tagger.errors import (
    DiscoveryError,
    PhotoTaggerError,
)
from photo_tagger.vocabulary_build import (
    KeywordCensus,
    census_from_export,
    census_from_photos,
    render_drop_report,
    render_vocabulary,
    trim,
    vocabulary_header,
)
from photo_tagger.vocabulary_organize import OrganizeStats, organize


def _census_source(inputs: list[Path] | None, from_export: Path | None) -> str:
    """Name where a vocabulary census came from, for telemetry: photos, an export, or both."""
    if inputs and from_export:
        return "both"
    return "export" if from_export else "photos"


def _build_census(
    inputs: list[Path] | None,
    from_export: Path | None,
    *,
    image_extensions: str,
    recursive: bool,
) -> tuple[KeywordCensus, str]:
    """Count keywords from the photos, from an export, or from both; also name the source."""
    census = KeywordCensus()
    sources: list[str] = []
    if from_export is not None:
        # utf-8-sig: a keyword export saved by a Windows editor starts with a BOM, which would
        # otherwise be counted as part of the first keyword.
        census.merge(census_from_export(from_export.read_text(encoding="utf-8-sig")))
        sources.append(f"keyword export {from_export.name} (counts are tree occurrences)")
    if inputs:
        image_files = resolve_image_batch(inputs, image_extensions, recursive=recursive)
        photo_census = census_from_photos(image_files)
        census.merge(photo_census)
        sources.append(f"{photo_census.photos} photo(s)")
    return census, " and ".join(sources)


def vocabulary(  # noqa: PLR0913 - every parameter is a separate concern of the command.
    inputs: Annotated[
        list[Path] | None,
        Parameter(
            name=("--input", "-i"),
            help="Photos or folders to read existing keywords from (repeat this option)",
        ),
    ] = None,
    *,
    output: Annotated[
        Path,
        Parameter(
            name=("--output", "-o"),
            validator=validators.Path(file_okay=True, dir_okay=False),
            help="Where to write the vocabulary file",
        ),
    ],
    from_export: Annotated[
        Path | None,
        Parameter(
            name=("--from-export",),
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
            help=(
                "Read a Lightroom keyword export (.txt or .csv) instead of, or as well as, the "
                "photos. Counts from an export are tree occurrences, not photos"
            ),
        ),
    ] = None,
    image_extensions: Annotated[
        str,
        Parameter(name=("--ext", "--extensions"), help="Extensions to scan for (case insensitive)"),
    ] = DEFAULT_EXTENSIONS,
    recursive: Annotated[
        bool,
        Parameter(name=("--recursive", "-r"), help="Scan subdirectories too"),
    ] = DEFAULT_RECURSIVE,
    build: Annotated[VocabularyBuildConfig, Parameter(name="*")] = VocabularyBuildConfig(),  # noqa: B008
    provider: Annotated[ProviderConfig, Parameter(name="*")] = DEFAULT_PROVIDER_OPTIONS,
) -> None:
    """
    Build a controlled vocabulary from the keywords a library already uses.

    ``--vocabulary-strict`` is what stops a catalog sprawling, and it needs a keyword file worth
    enforcing. This writes one: point it at your photos and it reads the keywords they already carry
    (through exiftool, so any application that writes XMP or IPTC works, not only Lightroom), counts
    how often each is used, and keeps the ones that earn their place.

    ``--from-export`` reads a Lightroom keyword export instead, for a catalog that is not on this
    machine. Its counts are occurrences in the keyword tree rather than photos, which is a weaker
    signal; prefer the photos when you have them.

    ``--organize`` adds a model pass over the keywords that survived, for the two things counting
    cannot settle: which of them are synonyms of each other, and what hierarchy they should have. It
    never decides what to keep and never invents a keyword.

    Nothing is written to your photos or your catalog. The output is a file to review and edit, plus
    an optional ``--report`` naming every keyword that was dropped and why.
    """
    export_exiftool_path()
    if not inputs and from_export is None:
        logger.error("vocabulary_no_source")
        raise SystemExit(1)

    console = Console()
    rules = to_trim_rules(build)
    try:
        census, source = _build_census(
            inputs,
            from_export,
            image_extensions=image_extensions,
            recursive=recursive,
        )
    except (DiscoveryError, OSError, UnicodeDecodeError) as exc:
        # UnicodeDecodeError is a ValueError, not an OSError: a keyword export saved as UTF-16
        # (or a binary file chosen by mistake) is a bad input, not a crash. --vocabulary has
        # always said so through VocabularyError; this says it the same way.
        logger.error("vocabulary_source_unreadable", error=str(exc))
        console.print(f"[red]Could not read the keyword source: {exc}[/red]")
        raise SystemExit(1) from exc

    if not census.uses:
        console.print("[yellow]No keywords found: nothing to build a vocabulary from.[/yellow]")
        raise SystemExit(1)

    result = trim(census, rules)
    telemetry.note(
        source=_census_source(inputs, from_export),
        keywords_read=len(census.uses),
        keywords_kept=len(result.kept),
        keywords_dropped=len(result.dropped),
    )
    stats: OrganizeStats | None = None
    if build.organize:
        console.print(f"Organizing {len(result.kept)} keyword(s) with {provider.model_name}...")
        try:
            result, stats = organize(
                result,
                provider_name=provider.provider_name,
                model_name=provider.model_name,
                api_base_url=provider.api_base_url,
                api_key=provider.api_key,
                workers=build.organize_workers,
            )
        except PhotoTaggerError as exc:
            # The trimmed list is already worth writing, but silently downgrading to it would hide
            # that the organize pass the user asked for never ran.
            logger.error("vocabulary_organize_failed", error=str(exc))
            console.print(f"[red]{exc}[/red]")
            raise SystemExit(1) from exc

    header = vocabulary_header(source, len(result.kept), len(result.dropped), rules, stats)
    try:
        output.write_text(
            render_vocabulary(result, header=header, flat=build.flat),
            encoding="utf-8",
        )
        if build.report_file is not None:
            build.report_file.write_text(render_drop_report(result), encoding="utf-8")
    except OSError as exc:
        logger.error("vocabulary_write_failed", error=str(exc))
        raise SystemExit(1) from exc

    console.print(f"Read {len(census.uses)} keyword(s) from {source}.")
    console.print(f"[green]Wrote {len(result.kept)} keyword(s) to {output}[/green]")
    if build.report_file is not None:
        console.print(f"Dropped {len(result.dropped)}; see {build.report_file}")
    console.print(
        "\nReview the file, then tag with it:\n"
        f"  photo-tagger -i PHOTOS --vocabulary {output} --vocabulary-strict",
    )
