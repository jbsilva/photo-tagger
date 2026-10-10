# Security Policy

photo-tagger is a local-first command-line tool (it runs on your own machine). I maintain it alone,
in my spare time. It reads your photos and asks a vision-language model that you choose to describe
them. Then it writes the results back as Lightroom-compatible metadata. This document explains how
to report a security problem and what I can realistically promise in return. It also shows where the
line is between bugs in this tool and the choices that you make as you run it.

This is a solo, unpaid, hobby project under the MIT license. There is no company, no security team,
no service-level agreement, and no bug bounty. I will still take security reports seriously and do
my best to fix real problems quickly.

## Supported Versions

photo-tagger is pre-1.0 beta software and follows Semantic Versioning. To keep maintenance
manageable, I ship security fixes only for the latest released line. Older releases do not get
backports (fixes copied to an older line). When a fix is ready, I ship it as a new release on that
line. To get the fix, upgrade.

| Version         | Supported          |
| --------------- | ------------------ |
| 0.10.0 (latest) | :white_check_mark: |
| < 0.10.0        | :x:                |

The current release is 0.10.0. Upgrade with:

```bash
uv tool upgrade photo-tagger
```

## Reporting a Vulnerability

Please report security problems privately. Do not open a public issue, pull request, or Discussion
for a vulnerability. A public report discloses the problem before there is a fix.

Use one of these channels:

1. Preferred: GitHub Private Vulnerability Reporting. Go to the **Security** tab of the repository.
   Click **Report a vulnerability**
   (<https://github.com/jbsilva/photo-tagger/security/advisories/new>). This opens a private draft
   advisory. There, you and I can work through the details together, out of public view. If the
   problem needs a public advisory, we can publish a coordinated advisory later.
2. Fallback: email. If private reporting is unavailable, or if you prefer email, write to
   python@juliobs.com. This is my general maintainer address, not a staffed security inbox. If you
   can, please use the GitHub channel.

I support coordinated (responsible) disclosure. First, report the problem to me privately. Then give
me a reasonable chance to fix it. After a fix is available, we go public together.

### What to Include in a Report

A clear report lets me assess the problem fast and prevents a slow back-and-forth. Please include as
much of this information as you can:

- The photo-tagger version (`photo-tagger --version`) and how you installed it.
- Your OS and Python version.
- The model backend that you used (Ollama, LM Studio, llama.cpp, OpenAI, or another
  OpenAI-compatible endpoint through `--url`), and whether it was local or remote.
- Steps to reproduce, or a small proof of concept.
- The impact: what an attacker can read, write, corrupt, or leak.
- Relevant log excerpts. Please remove API keys, absolute paths, and any personal location data
  before you send them.

### Response and Disclosure Process

This is a one-person project. Everything here is best-effort, with no hard guarantees:

- I aim to acknowledge a report within a few business days (roughly 3 to 5). Delays can happen, so
  this is a target, not a promise.
- If the report is valid, I will work with you privately to make sure that the problem is real. Then
  I will prepare a fix and publish a new release on the supported line. After that, I will publish a
  GitHub Security Advisory. Where it makes sense, I will also request a CVE.
- I aim to coordinate public disclosure within about 90 days of the report. That window is a
  ceiling, not a countdown. If a fix is straightforward, we will move faster. If it is genuinely
  complex, I can ask to extend the embargo (the private period before disclosure). I prefer a
  correct fix to a rushed one.
- I am happy to credit you in the advisory and the release notes. Tell me how you want me to name
  you. If you ask, I will keep you anonymous.

## Scope and Threat Model

The most important thing to understand about photo-tagger is that it is local-first. I never receive
your images or your metadata. The tool sends your photos only to the model endpoint you configure.
By default, that endpoint is localhost: Ollama at `http://localhost:11434/v1` or LM Studio at
`http://localhost:1234/v1`. If you turn on `--otel`, it also sends traces to an OpenTelemetry
collector that you name. Its one other network call is the anonymous usage telemetry that
[docs/telemetry.md](docs/telemetry.md) describes. It sends names, counts, durations, and switches,
never a path, a photo, or text from one. It goes to a small write-only collector (the Worker in
[`telemetry/`](telemetry/)), and you can turn it off.

Because of that design, the biggest real-world risks are configuration choices, not code defects. A
vulnerability report must be about a flaw in photo-tagger's own code, not about how you pointed it.

### In scope

Bugs in photo-tagger's own behavior are in scope, for example:

- Path handling: the tool derives XMP sidecar paths and writes summary, skip-list, cache, lock, and
  log files. Writing outside the intended location or clobbering an unrelated file is a bug.
- Unsafe handling of malformed or malicious image input: the tool decodes arbitrary RAW and standard
  images from disk. It must handle a crafted file gracefully. It must not let an unhandled crash
  take down a whole batch, beyond what the upstream decoders do.
- Secret leakage in artifacts: API keys must not end up in logs, the SQLite cache, the summary JSON,
  or the NDJSON output. Today only `api_key_present=true/false` is logged. Provider error bodies are
  truncated before logging, so an echoed `Authorization` header is not dumped in full. A regression
  that leaks a key is in scope.
- File permissions and atomicity for artifacts the tool controls: the lock file is created with mode
  `0o600`, and the lock releases on crash. The summary file is written atomically. Lock leaks,
  world-readable files the tool creates, or partial/corrupt writes are in scope.
- SQLite cache safety: the cache uses parameterized queries and serializes access behind a lock. SQL
  injection or cache-corruption bugs introduced here are in scope.
- Endpoint URL validation: listing probes reject non-`http(s)` schemes and missing hosts. Keeping
  that validation correct is in scope.
- Concurrency: with `--workers > 1`, shared resources (cache, skip list, per-thread ExifTool
  helpers) are lock-protected. Data races that corrupt those files are in scope.
- Supply-chain integrity of what I ship: SHA-pinned GitHub Actions, a committed `uv.lock`, Renovate
  with a release cooldown, and PyPI publishing via OIDC trusted publishing. Keeping the release
  pipeline sound is in scope.
- What leaves the machine in a request: the prompt forwards a curated subset of metadata (existing
  keywords, City/Country, GPS position, camera model, lens, capture date). Sending more than that,
  or adding fields by accident, is in scope.
- What the telemetry sends and keeps: a beacon must carry only the fields that
  [docs/telemetry.md](docs/telemetry.md) lists, and the collector must not store IP addresses. A
  beacon that carries a path, text from a photo, or anything else that identifies you is in scope.
  So is a flaw in the collector code in [`telemetry/`](telemetry/).

### Out of scope

These are not vulnerabilities in photo-tagger:

- Your model server: the security, patch level, auth, and TLS of your Ollama / LM Studio / llama.cpp
  / OpenAI-compatible server are yours to manage. photo-tagger is only a client.
- Where you point `--url`: the tool sends the image and metadata to whatever endpoint you configure.
  Pointing it at a remote or untrusted host, and any data exposure that follows, is your decision.
  If you point `--url` off localhost, the tool does not warn you. It also permits plain `http://`
  there, so a remote host can receive your data as plaintext.
- ExifTool, libraw/rawpy, and Pillow CVEs: the tool shells out to the ExifTool binary on your PATH
  (or at the path that you configure) and uses third-party decoders. Flaws in those tools are
  upstream. My job is to invoke them safely and keep dependency pins current, not to fix the
  libraries.
- The model's output content: a model can hallucinate or produce wrong or offensive titles,
  descriptions, and keywords. The tool makes sure that output matches a schema, and keyword counts
  are capped. I cannot guarantee semantic correctness.
- Prompt injection via image or EXIF content: model output is only written into metadata fields. It
  is never executed. It is never used to build shell commands or filesystem paths. A model coaxed
  into emitting odd text is a content issue, not an injection vulnerability.
- Third-party Python dependency CVEs: tracked via Renovate and CodeQL, but the underlying flaws are
  upstream. My duty is to ship updated pins, not to patch the libraries.
- Secrets exposed because you passed `--api-key` on the command line: command-line arguments are
  visible to other local users in process listings. This is documented, and environment variables
  are recommended instead. The OS-level visibility is not something the tool can fix.
- The platform under the collector: the telemetry collector runs on Cloudflare Workers. The security
  of Cloudflare itself is upstream.

## Security Considerations for Users

Most of your real risk comes from how you run the tool, not from the tool's code. A few habits keep
you safe:

- Prefer environment variables over `--api-key`: use `OLLAMA_API_KEY`, `LM_STUDIO_API_KEY`,
  `LLAMA_CPP_API_KEY`, or `OPENAI_API_KEY`. Command-line arguments show up in process listings (for
  example, `ps`) to other local users. The `--api-key` help text warns about this.
- Be deliberate about `--url` and `*_BASE_URL` targets: keep them on localhost for fully offline
  processing. Anything else sends your image content, and any embedded GPS/location and camera
  metadata, to that host. Only use endpoints you trust, and prefer `https://` for remote ones. If
  you tell the tool to use plain `http`, it will transmit your data that way.
- Mind your GPS and location tags: the prompt surfaces GPS position and City/Country as context. If
  you do not want coordinates leaving your machine, process locally, or strip those tags with
  ExifTool before running against a remote endpoint.
- Protect the on-disk artifacts: log files (`./logs` by default), the SQLite cache (`--cache-file`),
  the summary JSON (`--summary-file`), and skip lists can contain file paths, generated descriptions
  and keywords, and read location data. Keep them in a directory only you can read. If you do not
  need an audit trail, turn off file logging with `--file-log-level OFF`.
- Keep ExifTool, libraw, and the Python dependencies current: patching those upstream tools is how
  you stay ahead of image-parsing and metadata CVEs. Reinstall or upgrade photo-tagger to pick up
  bumped pins.
- Try new setups safely: when you test a new endpoint or prompt, use `--dry-run` to review the
  proposed metadata before it changes any file. Rely on the default XMP-sidecar behavior (avoid
  `--embed-in-photo` and `--no-backup-xmp`) so originals stay recoverable.
- Use `--lock-file` for batches, so two runs cannot race on the same folder. The lock file is
  created with mode `0o600`. If the process crashes, the lock releases automatically.
- Secure your model server yourself: an Ollama or LM Studio instance you expose on the network is
  its own attack surface. Bind it to localhost or protect it with auth and network controls.
  photo-tagger does not secure your model server for you.

## Safe Harbor

I welcome good-faith security research on photo-tagger. If you act in good faith and follow this
policy, I will not pursue or support legal action against you for your research. Please keep your
testing on your own machine and your own photos. Avoid privacy violations and any disruption of
other people's systems or services. Give me a reasonable chance to fix an issue before you disclose
it publicly. Thank you for helping keep the project and its users safe.
