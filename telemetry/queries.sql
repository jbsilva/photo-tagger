-- Starter queries for the photo-tagger telemetry dataset.
--
-- Run these against the Cloudflare Analytics Engine SQL API (or paste into the dashboard's SQL
-- console). AE speaks a subset of the ClickHouse SQL dialect. AE keeps three months of data; the
-- D1 table at the end of this file keeps everything.
--
-- Every count is SUM(_sample_interval) rather than COUNT(): once volume is high enough that AE
-- starts sampling, _sample_interval is the per-row weight (it is 1 while no sampling happens), so
-- weighting keeps the totals honest. The same reason is why the medians use quantileWeighted.
--
-- Column map (see worker.js). One dataset holds every event kind, discriminated by blob12:
--   run rows     -> WHERE blob12 IN ('run', '')   (v1 rows may have '' there)
--   crash rows   -> WHERE blob12 = 'crash'
--   v3 kinds     -> 'session' (a GUI window, a watch), 'command' (the other commands), 'launch'
--   v2+ metrics (outcome counters, hardware) -> add AND double1 >= 2
-- GUI run rows from v1/v2 clients are whole window sessions (one per close, idle time included);
-- from v3 on, a GUI run row is one generation or save, and the window sends a 'session' row.
--
--   index1  = install_id
--   blob1   = app_version         blob2   = interface (cli|gui)  blob3   = provider
--   blob4   = model               blob5   = arch                 blob6   = os
--   blob7   = os_release          blob8   = python_version       blob9   = output_language
--   blob10  = ui_language         blob11  = file_types           blob12  = event
--   blob13  = cpu model           blob14  = gpu model            blob15  = exception_type
--   blob16  = crash_location      blob17  = crash_frames         blob18  = failure_kinds
--   blob19  = the whole beacon as JSON (v3: settings, job, session, and setup facts)
--   double1 = schema_version      double2 = batch_size           double3 = duration_seconds
--   double4 = success_count       double5 = failure_count        double6 = cache_hits
--   double7 = retry_successes     double8 = workers              double9 = total_tokens
--   double10 = inference_seconds  double11 = cpu_count           double12 = memory_gb
--   double13 = dry_run (0|1)


-- 1. Most-used models (histogram), last 30 days.
SELECT blob4 AS model, SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 IN ('run', '')
GROUP BY model
ORDER BY runs DESC;


-- 2. Typical batch size: the median number of photos per run.
SELECT quantileWeighted(0.5)(double2, _sample_interval) AS median_batch_size
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 IN ('run', '');


-- 3. CLI vs GUI split.
SELECT blob2 AS interface, SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 IN ('run', '')
GROUP BY interface
ORDER BY runs DESC;


-- 4. Performance by CPU model: median run duration per chip. A CLI run and a v3 GUI run both time
-- the work itself; a v1/v2 GUI row timed the whole window, idle time included, so it is left out.
SELECT
  blob13 AS cpu,
  quantileWeighted(0.5)(double3, _sample_interval) AS median_seconds,
  SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY
  AND blob12 IN ('run', '') AND (blob2 = 'cli' OR double1 >= 3) AND blob13 != ''
GROUP BY cpu
ORDER BY runs DESC;


-- 5. OS distribution.
SELECT blob6 AS os, SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 IN ('run', '')
GROUP BY os
ORDER BY runs DESC;


-- 6a. Active installs (distinct install ids) over the last 30 days.
SELECT COUNT(DISTINCT index1) AS active_installs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY;

-- 6b. Active installs per day, for a trend line over the last 90 days.
SELECT
  toStartOfInterval(timestamp, INTERVAL '1' DAY) AS day,
  COUNT(DISTINCT index1) AS active_installs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '90' DAY
GROUP BY day
ORDER BY day;


-- 7. File-type combinations per run. blob11 is the sorted set of extensions in one batch (e.g.
-- "cr3,jpg"), so this groups by the whole combination. AE SQL has no splitByChar to explode it into
-- one row per format; the dashboard does that split client-side. This still shows RAW vs JPEG usage.
SELECT blob11 AS file_types, SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 IN ('run', '') AND blob11 != ''
GROUP BY file_types
ORDER BY runs DESC;


-- 8. Metadata (output) language vs UI language.
SELECT blob9 AS output_language, blob10 AS ui_language, SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 IN ('run', '')
GROUP BY output_language, ui_language
ORDER BY runs DESC;


-- 9. GPU distribution (schema v2): which graphics hardware is out there?
SELECT blob14 AS gpu, SUM(_sample_interval) AS runs, COUNT(DISTINCT index1) AS installs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 IN ('run', '') AND blob14 != ''
GROUP BY gpu
ORDER BY runs DESC;


-- 10. CPU model distribution (schema v2).
SELECT blob13 AS cpu, SUM(_sample_interval) AS runs, COUNT(DISTINCT index1) AS installs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 IN ('run', '') AND blob13 != ''
GROUP BY cpu
ORDER BY runs DESC;


-- 11. RAM distribution (schema v2). memory_gb is already a whole number of gigabytes, so raw
-- grouping clusters naturally (8, 16, 18, 24, 32, 36, 64, ...).
SELECT double12 AS memory_gb, SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 IN ('run', '') AND double12 > 0
GROUP BY memory_gb
ORDER BY memory_gb;


-- 12. Success and cache-hit rates (schema v2, real writes only: dry runs excluded). Divide
-- client-side: ok / (ok + failed), and cache_hits / photos.
SELECT
  SUM(double4 * _sample_interval) AS ok,
  SUM(double5 * _sample_interval) AS failed,
  SUM(double6 * _sample_interval) AS cache_hits,
  SUM(double2 * _sample_interval) AS photos
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY
  AND blob12 IN ('run', '') AND double1 >= 2 AND double13 = 0;


-- 13. Token appetite: median tokens per run, and total (schema v2).
SELECT
  quantileWeighted(0.5)(double9, _sample_interval) AS median_tokens_per_run,
  SUM(double9 * _sample_interval) AS total_tokens
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 IN ('run', '') AND double9 > 0;


-- 13b. Why photos fail: per-run failure buckets, e.g. "timeout:3,metadata-write:1". Like
-- file_types, the compact set cannot be exploded in AE SQL; the dashboard splits it client-side.
SELECT blob18 AS failure_kinds, SUM(_sample_interval) AS runs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 IN ('run', '') AND blob18 != ''
GROUP BY failure_kinds
ORDER BY runs DESC;


-- 14a. Crash count and affected installs, last 30 days.
SELECT SUM(_sample_interval) AS crashes, COUNT(DISTINCT index1) AS affected_installs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 = 'crash';

-- 14b. Top crash signatures: what breaks, and where in the code.
SELECT
  blob15 AS exception_type,
  blob16 AS crash_location,
  blob1 AS app_version,
  SUM(_sample_interval) AS crashes,
  COUNT(DISTINCT index1) AS installs
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 = 'crash'
GROUP BY exception_type, crash_location, app_version
ORDER BY crashes DESC;

-- 14c. Crashes per day, for a trend line.
SELECT
  toStartOfInterval(timestamp, INTERVAL '1' DAY) AS day,
  SUM(_sample_interval) AS crashes
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '90' DAY AND blob12 = 'crash'
GROUP BY day
ORDER BY day;

-- 14d. Crashes by app version: is the newest release healthier?
SELECT blob1 AS app_version, SUM(_sample_interval) AS crashes
FROM photo_tagger_telemetry
WHERE timestamp > NOW() - INTERVAL '30' DAY AND blob12 = 'crash'
GROUP BY app_version
ORDER BY crashes DESC;


-- 15. D1: the long-term table (see README.md, "What it stores"). Run with
--   wrangler d1 execute photo-tagger-telemetry --remote --command "..."
-- or through the dashboard's SQL console. payload is the beacon as JSON; read a field with ->>.
-- Settings popularity, all time: which sidecar mode do runs use?
--   SELECT payload ->> '$.settings.sidecar_mode' AS sidecar_mode, COUNT(*) AS runs
--   FROM events
--   WHERE event = 'run'
--   GROUP BY sidecar_mode
--   ORDER BY runs DESC;
