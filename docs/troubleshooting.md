---
icon: lucide/wrench
---

# Troubleshooting

The three failures below are the cause of almost every run that will not start. Each entry shows the
symptom first, then the fix. If you are not sure of the cause, run the built-in pre-flight check. It
tests exactly the two external parts (ExifTool and the model server) that cause these failures:

```bash
photo-tagger doctor
```

## "exiftool not found"

Symptom: the run (or `doctor`) fails immediately with a message that `exiftool` was not found.

photo-tagger runs the [ExifTool](https://exiftool.org/) binary for all metadata reads and writes.
photo-tagger does not include ExifTool. Install it with your system package manager:

```bash
brew install exiftool                  # macOS (Homebrew)
apt install libimage-exiftool-perl    # Debian / Ubuntu
```

As an alternative, download a build for your platform from [exiftool.org](https://exiftool.org/).
For more package managers, see [Installation](getting-started/installation.md#system-requirements).

After the installation, photo-tagger must find the binary. The binary must be on your `PATH`, or you
must give its full path. To make sure that it is on your `PATH`, run `exiftool -ver`.

If the binary is in an unusual location, set the `exiftool_path` key in your
[configuration file](getting-started/configuration.md#config-file) to its full path. You can also
set the `PHOTO_TAGGER_EXIFTOOL` environment variable to that path. If you set both, the environment
variable wins.

!!! note

    Finder on macOS gives apps a minimal `PATH`. If you start the desktop GUI from Finder, the GUI
    automatically uses the same `PATH` as your login shell. As a result, it finds an ExifTool from
    Homebrew or MacPorts.

## Connection refused / could not list models

Symptom: the run fails with a connection error, or `doctor` reports that it was unable to reach the
provider or list its models.

photo-tagger needs a model server that is running. Answer these questions in order:

1. Is the server actually running? Start Ollama, LM Studio's server, or `llama-server` first. The
    `openai` provider talks to a hosted API. For that provider, make sure that your network and
    your API key work instead.

2. Is it on the port that photo-tagger expects? Each provider has a default base URL:

    | Provider  | Default base URL            |
    | --------- | --------------------------- |
    | Ollama    | `http://localhost:11434/v1` |
    | LM Studio | `http://localhost:1234/v1`  |
    | llama.cpp | `http://localhost:8080/v1`  |
    | OpenAI    | `https://api.openai.com/v1` |

3. Do you need a different URL? If your server listens somewhere else (a different port, a remote
    host), pass `-u/--url`. You can also set the environment variable of the provider instead:
    `OLLAMA_BASE_URL`, `LM_STUDIO_BASE_URL`, `LLAMA_CPP_BASE_URL`, or `OPENAI_BASE_URL`.

Then run `photo-tagger doctor --provider <name>` to make sure that the fix works.

## "Model 'X' not available"

Symptom: the provider is reachable, but the run stops with a message that the requested model is not
available on it.

Before photo-tagger processes any photo, it compares the `-m/--model` identifier with the models
that the server actually serves. As a result, a typo or a model that you never pulled makes the run
fail at the start. To fix it:

- List what the server serves. If the check fails, `photo-tagger doctor` prints the available
    models. In the GUI, press the **Refresh** button next to the model dropdown.
- Pull or load the model first. For Ollama, run `ollama pull <model>`. For LM Studio, download the
    model in its UI. Then load it into the server. For llama.cpp, start `llama-server` with the
    model file.
- Match the identifier exactly. The identifier must match the spelling on the server. This includes
    any namespace or tag (for example `qwen/qwen3-vl-30b` or `llava:34b`).
