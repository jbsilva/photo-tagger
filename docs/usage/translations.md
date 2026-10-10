---
icon: lucide/languages
---

# Translations

The desktop GUI is translated at runtime with GNU gettext. English is the source language. Brazilian
Portuguese (`pt_BR`) ships complete. The CLI's `--help` text and the structured log events
intentionally stay in English so scripts and bug reports remain grep-able.

!!! note

    This page is about the language of the interface. The language of the generated metadata (titles,
    descriptions, keywords) is a separate, free-form setting: **Settings > Metadata Language** in the
    GUI, [`--output-language`](cli-reference.md#inference) in the CLI, or `output_language` under
    `[inference]` in the [configuration file](../getting-started/configuration.md).

## Choosing a language

By default, the GUI follows the operating system's language. When there is no matching translation,
it falls back to English. To pick one explicitly, use any of these (first match wins):

1. The `PHOTO_TAGGER_LANG` environment variable, for a one-off run:

    ```bash
    PHOTO_TAGGER_LANG=pt_BR photo-tagger gui
    ```

2. **Settings > Language** in the GUI. The choice is written to the configuration file (comments and
    the rest of the configuration stay as they are), and it applies on the next launch. **System
    Default** removes it again.

3. The top-level `language` key in the [configuration file](../getting-started/configuration.md):

    ```toml
    language = "pt_BR"
    ```

Accepted values are `en`, `pt_BR` (any common spelling works: `pt-BR`, `pt_BR.UTF-8`, plain `pt`),
or `auto` for the OS default. Unknown values quietly fall back to English.

## Contributing a translation

Catalogs live in `src/photo_tagger/locale/`. The editable `.po` files are committed next to the
compiled `.mo` files gettext loads. To add a language (say German):

```bash
# 1. Create the catalog from the current template
uv run pybabel init -i src/photo_tagger/locale/photo_tagger.pot \
    -d src/photo_tagger/locale -D photo_tagger -l de

# 2. Fill in the msgstr entries in src/photo_tagger/locale/de/LC_MESSAGES/photo_tagger.po

# 3. Compile the runtime catalog (a pre-commit hook fails if you forget)
uv run scripts/compile_translations.py
```

Then add the language code and its native-language label to `SUPPORTED_LANGUAGES` in
`src/photo_tagger/i18n.py`, so it appears in the GUI's Language menu. Add a catalog-loading test to
`tests/test_i18n.py`.

After changing translatable strings in the source, refresh the template and merge it into the
existing catalogs. The script wraps `pybabel extract` + `pybabel update`. It also fills in the
catalog header metadata (copyright holder, bug-report address, translator) that the bare commands
leave as placeholders:

```bash
uv run poe translations    # Runs extract_translations.py, then compile_translations.py
```
