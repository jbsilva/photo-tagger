# Security Policy

photo-tagger is a local-first command-line tool maintained by one person in their spare time. It
reads your photos, asks a vision-language model you choose to describe them, and writes the results
back as Lightroom-compatible metadata. This document explains how to report a security problem, what
I can realistically promise in return, and where the line sits between bugs in this tool and choices
you make when you run it.

This is a solo, unpaid, hobby project under the MIT license. There is no company, no security team,
no service-level agreement, and no bug bounty. I will still take security reports seriously and do
my best to fix real problems quickly.

## Supported Versions

photo-tagger is pre-1.0 beta software and follows Semantic Versioning. To keep maintenance sane, I
only ship security fixes for the latest released line. Older releases do not get backports. When a
fix lands, it goes out as a new release on that line, so upgrading is how you get the fix.

| Version        | Supported          |
| -------------- | ------------------ |
| 0.2.2 (latest) | :white_check_mark: |
| < 0.2.2        | :x:                |

The current release is 0.2.2. Upgrade with:

```bash
uv tool install --upgrade photo-tagger
```

## Reporting a Vulnerability

Please report security problems privately. Do not open a public issue, pull request, or Discussion
for a vulnerability, because that discloses the problem before there is a fix.

Use one of these channels:

1. **Preferred: GitHub Private Vulnerability Reporting.** Go to the repository's **Security** tab
   and click **Report a vulnerability**
   (<https://github.com/jbsilva/photo-tagger/security/advisories/new>). This opens a private draft
   advisory where you and I can work through the details together, out of public view, and later
   publish a coordinated advisory if one is warranted.
2. **Fallback: email.** If private reporting is unavailable or you would rather use email, write to
   **python@juliobs.com**. This is my general maintainer address, not a staffed security inbox, so
   please use the GitHub channel when you can.

I support coordinated (responsible) disclosure: report it to me privately first, give me a
reasonable chance to fix it, and we go public together once a fix is available.

### What to Include in a Report

A clear report lets me triage fast and avoids slow back-and-forth. Please include as much of the
following as you can:

- The photo-tagger version (`photo-tagger --version`) and how you installed it.
- Your OS and Python version.
- The model backend in play (Ollama, LM Studio, or another OpenAI-compatible endpoint via `--url`),
  and whether it was local or remote.
- Steps to reproduce, or a small proof of concept.
- The impact: what an attacker could read, write, corrupt, or leak.
- Relevant log excerpts. Please redact API keys, absolute paths, and any personal location data
  before sending.

### Response and Disclosure Process

This is a one-person project, so everything here is best-effort with no hard guarantees:

- I aim to acknowledge a report within a few business days (roughly 3 to 5). Life happens, so this
  is a target, not a promise.
- If the report is valid, I will work with you privately to confirm it, prepare a fix, cut a release
  on the supported line, and then publish a GitHub Security Advisory (and request a CVE where it
  makes sense).
- I aim to coordinate public disclosure within about 90 days of the report. That window is a
  ceiling, not a countdown: if a fix is straightforward we will move faster, and if it is genuinely
  complex I may ask to extend the embargo. I would rather ship a correct fix than a rushed one.
- I am happy to credit you in the advisory and release notes. Tell me how you would like to be
  named, or say the word and I will keep you anonymous.

## Scope and Threat Model

The most important thing to understand about photo-tagger is that it is **local-first** and I run no
hosted service. I never receive your images, your metadata, or any telemetry. There is no backend to
attack. The tool only makes network calls to the model endpoint **you** configure, and by default
that is localhost (Ollama at `http://localhost:11434/v1` or LM Studio at
`http://localhost:1234/v1`).

Because of that design, the biggest real-world risks are configuration choices, not code defects. A
vulnerability report should be about a flaw in photo-tagger's own code, not about how you pointed
it.

### In scope

Bugs in photo-tagger's own behavior are in scope, for example:

- **Path handling.** The tool derives XMP sidecar paths and writes summary, skip-list, cache, lock,
  and log files. Writing outside the intended location or clobbering an unrelated file is a bug.
- **Unsafe handling of malformed or malicious image input.** The tool decodes arbitrary RAW and
  standard images from disk. A crafted file should be handled gracefully and must not, for instance,
  cause an unhandled crash that takes down a whole batch, beyond what the upstream decoders do.
- **Secret leakage in artifacts.** API keys must not end up in logs, the SQLite cache, the summary
  JSON, or the NDJSON output. (Today only `api_key_present=true/false` is logged, and provider error
  bodies are truncated before logging so an echoed `Authorization` header is not dumped in full.) A
  regression that leaks a key is in scope.
- **File permissions and atomicity for artifacts the tool controls.** The lock file is created with
  mode `0o600`, the lock releases on crash, and the summary file is written atomically. Lock leaks,
  world-readable files the tool creates, or partial/corrupt writes are in scope.
- **SQLite cache safety.** The cache uses parameterized queries and serializes access behind a lock.
  SQL injection or cache-corruption bugs introduced here are in scope.
- **Endpoint URL validation.** Listing probes reject non-`http(s)` schemes and missing hosts.
  Keeping that check correct is in scope.
- **Concurrency.** With `--workers > 1`, shared resources (cache, skip list, per-thread ExifTool
  helpers) are lock-protected. Data races that corrupt those files are in scope.
- **Supply-chain integrity of what I ship.** SHA-pinned GitHub Actions, a committed `uv.lock`,
  Dependabot with a release cooldown, and PyPI publishing via OIDC trusted publishing. Keeping the
  release pipeline sound is in scope.
- **What leaves the machine in a request.** The prompt forwards a curated subset of metadata
  (existing keywords, City/Country, GPS position, camera model, lens, capture date). Sending more
  than that, or adding fields by accident, is in scope.

### Out of scope

These are not vulnerabilities in photo-tagger:

- **Your model server.** The security, patch level, auth, and TLS of your Ollama / LM Studio /
  OpenAI-compatible server are yours to manage. photo-tagger is only a client.
- **Where you point `--url`.** The tool faithfully sends the image and metadata to whatever endpoint
  you configure. Pointing it at a remote or untrusted host, and any data exposure that follows, is
  your decision. Note the tool permits plain `http://` and does not warn when the endpoint is not on
  localhost, so plaintext transmission to a remote host is possible if you set it up that way.
- **ExifTool, libraw/rawpy, and Pillow CVEs.** The tool shells out to the ExifTool binary on your
  PATH and uses third-party decoders. Flaws in those tools are upstream. My job is to invoke them
  safely and keep dependency pins current, not to fix the libraries.
- **The model's output content.** A model can hallucinate or produce wrong or offensive titles,
  descriptions, and keywords. Output is validated against a schema and keyword counts are capped,
  but semantic correctness is not something I can guarantee.
- **Prompt injection via image or EXIF content.** Model output is only written into metadata fields.
  It is never executed and never used to build shell commands or filesystem paths, so a model coaxed
  into emitting odd text is a content issue, not an injection vulnerability.
- **Third-party Python dependency CVEs.** Tracked via Dependabot and CodeQL, but the underlying
  flaws are upstream. My duty is to ship updated pins, not to patch the libraries.
- **Secrets exposed because you passed `--api-key` on the command line.** Command-line arguments are
  visible to other local users in process listings. This is documented, and environment variables
  are recommended instead. The OS-level visibility is not something the tool can fix.
- **Any hosted service or telemetry.** There is none, so there is no server-side attack surface.

## Security Considerations for Users

Most of your real risk comes from how you run the tool, not from the tool's code. A few habits keep
you safe:

- **Prefer environment variables over `--api-key`.** Use `OLLAMA_API_KEY`, `LM_STUDIO_API_KEY`, or
  `OPENAI_API_KEY`. Command-line arguments show up in process listings (for example, `ps`) to other
  local users; the `--api-key` help text warns about this.
- **Be deliberate about `--url` and `*_BASE_URL` targets.** Keep them on localhost for fully offline
  processing. Anything else sends your image content, and any embedded GPS/location and camera
  metadata, to that host. Only use endpoints you trust, and prefer `https://` for remote ones, since
  the tool will transmit over plaintext `http` if you tell it to.
- **Mind your GPS and location tags.** The prompt surfaces GPS position and City/Country as context.
  If you do not want coordinates leaving your machine, process locally, or strip those tags with
  ExifTool before running against a remote endpoint.
- **Protect the on-disk artifacts.** Log files (`./logs` by default), the SQLite cache
  (`--cache-file`), the summary JSON (`--summary-file`), and skip lists can contain file paths,
  generated descriptions and keywords, and read location data. Keep them in a directory only you can
  read, or turn off file logging with `--file-log-level OFF` when you do not need an audit trail.
- **Keep ExifTool, libraw, and the Python dependencies current.** Patching those upstream tools is
  how you stay ahead of image-parsing and metadata CVEs. Reinstall or upgrade photo-tagger to pick
  up bumped pins.
- **Try new setups safely.** Use `--dry-run` when testing a new endpoint or prompt so you can review
  the proposed metadata before any file is touched, and rely on the default XMP-sidecar behavior
  (avoid `--embed-in-photo` and `--no-backup-xmp`) so originals stay recoverable.
- **Use `--lock-file` for batches** so two runs cannot race on the same folder. The lock file is
  created mode `0o600` and is released automatically if the process crashes.
- **Secure your model server yourself.** An Ollama or LM Studio instance you expose on the network
  is its own attack surface. Bind it to localhost or protect it with auth and network controls.
  photo-tagger does not secure your model server for you.

## Safe Harbor

I welcome good-faith security research on photo-tagger. If you act in good faith and follow this
policy, I will not pursue or support legal action against you for your research. Please keep your
testing on your own machine and your own photos, avoid privacy violations and any disruption of
other people's systems or services, and give me a reasonable chance to fix an issue before
disclosing it publicly. Thank you for helping keep the project and its users safe.
