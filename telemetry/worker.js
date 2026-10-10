/**
 * photo-tagger telemetry collector (Cloudflare Worker).
 *
 * Accepts one anonymous JSON beacon per event ("run", "crash", "session", "command", "launch")
 * and stores it twice:
 *   - Analytics Engine: the fixed column layout queries.sql reads (blob1..18, double1..13), plus
 *     the whole sanitized beacon as JSON in blob19. AE keeps data for three months.
 *   - D1, when the DB binding exists: one row per beacon, the same JSON included. D1 keeps
 *     everything, so the dashboard can look back further than three months.
 * The fields come from src/photo_tagger/telemetry.py and are listed in docs/telemetry.md.
 *
 * It is deliberately minimal and privacy-preserving:
 *   - Write-only: it never reads anything back and returns an empty 204.
 *   - It stores no IP address, sets no cookies, and writes only the fields the beacon carries.
 *   - A beacon is sanitized before it is stored: snake_case keys only, scalar values or one level
 *     of nesting, every string clamped, so a hostile client cannot bloat or reshape a row.
 *
 * Schema compatibility: v1 and v2 clients (older installs) keep working. New fields only ever
 * extend the layout, so old and new rows can be queried uniformly.
 */

const KNOWN_SCHEMAS = new Set([1, 2, 3]);
// Clamp for the AE text columns.
const MAX_STR = 200;
// Clamp for strings inside the JSON payload. Lists like the flags a run used can run long.
const MAX_JSON_STR = 1000;
// Top-level fields allowed past MAX_JSON_STR. A crash stack names up to 40 frames by module,
// function, and line, which is what makes a crash report worth reading.
const LONG_JSON_STR = { crash_stack: 6000 };
// A real beacon is a few KiB; anything bigger is not ours. Rejecting on Content-Length keeps a
// hostile client from making the Worker parse megabytes of JSON.
const MAX_BODY_BYTES = 32768;
// Caps the number of keys a beacon may carry, nested ones included.
const MAX_KEYS = 400;
// AE caps all the blobs of one data point at 16 KB together; leave room for blob1..18.
const MAX_BLOB_JSON_BYTES = 12000;
const KEY_PATTERN = /^[a-z][a-z0-9_]{0,63}$/;

// Coerce to a bounded string; anything non-string (or oversized) becomes a safe value.
const str = (v) => (typeof v === "string" ? v.slice(0, MAX_STR) : "");
// Coerce to a finite number; NaN/Infinity/non-numbers become 0. Booleans count as 0 or 1.
const num = (v) => {
  if (typeof v === "boolean") return v ? 1 : 0;
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
};

/**
 * Reduce one value to what may be stored: a clamped string, a finite number, a boolean, null, or
 * (at the top level only) an object of those. Anything else is dropped.
 */
function clean(value, nested, budget, limit = MAX_JSON_STR) {
  switch (typeof value) {
    case "string":
      return value.slice(0, limit);
    case "number":
      return Number.isFinite(value) ? value : 0;
    case "boolean":
      return value;
    case "object":
      if (value === null) return null;
      if (Array.isArray(value) || nested) return undefined;
      return cleanObject(value, true, budget);
    default:
      return undefined;
  }
}

function cleanObject(object, nested, budget) {
  const out = {};
  for (const [key, value] of Object.entries(object)) {
    if (budget.keys-- <= 0) break;
    if (!KEY_PATTERN.test(key)) continue;
    const limit = nested ? MAX_JSON_STR : (LONG_JSON_STR[key] ?? MAX_JSON_STR);
    const kept = clean(value, nested, budget, limit);
    if (kept !== undefined) out[key] = kept;
  }
  return out;
}

const sanitize = (body) => cleanObject(body, false, { keys: MAX_KEYS });

function writeAnalytics(env, body, json) {
  const fits = new TextEncoder().encode(json).length <= MAX_BLOB_JSON_BYTES;
  env.TELEMETRY.writeDataPoint({
    // index1: the random install id, so COUNT(DISTINCT index1) approximates active installs.
    indexes: [str(body.install_id)],
    // blob1..blob19, in the order queries.sql reads them. blob1..blob11 match schema v1 exactly.
    blobs: [
      str(body.app_version), //      blob1
      str(body.interface), //        blob2  "cli" | "gui"
      str(body.provider), //         blob3
      str(body.model), //            blob4
      str(body.arch), //             blob5
      str(body.os), //               blob6
      str(body.os_release), //       blob7
      str(body.python_version), //   blob8
      str(body.output_language), //  blob9  metadata language, e.g. "English"
      str(body.ui_language), //      blob10 resolved UI language code, e.g. "en"
      str(body.file_types), //       blob11 distinct extensions, e.g. "cr3,jpg"
      str(body.event) || "run", //   blob12 event name (v1 rows normalize to "run")
      str(body.cpu), //              blob13 CPU model, e.g. "Apple M3 Pro"
      str(body.gpu), //              blob14 GPU model, e.g. "NVIDIA GeForce RTX 4070"
      str(body.exception_type), //   blob15 crash only: exception class name
      str(body.crash_location), //   blob16 crash only: deepest in-app frame "module:func:line"
      str(body.crash_frames), //     blob17 crash only: last in-app frames, '>'-joined
      str(body.failure_kinds), //    blob18 run only: final-failure buckets "timeout:3,other:1"
      fits ? json : "", //           blob19 the whole sanitized beacon as JSON (v3+ fields)
    ],
    // double1..double13. double1..double3 match schema v1; the rest are v2 additions.
    doubles: [
      num(body.schema_version), //   double1
      num(body.batch_size), //       double2
      num(body.duration_seconds), // double3
      num(body.success_count), //    double4
      num(body.failure_count), //    double5
      num(body.cache_hits), //       double6
      num(body.retry_successes), //  double7
      num(body.workers), //          double8
      num(body.total_tokens), //     double9
      num(body.inference_seconds), //double10
      num(body.cpu_count), //        double11
      num(body.memory_gb), //        double12
      num(body.dry_run), //          double13 1 when the run previewed without writing
    ],
  });
}

function insertEvent(env, body, json) {
  return env.DB.prepare(
    "INSERT INTO events (received_at, install_id, session_id, event, interface, app_version, " +
      "schema_version, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
  )
    .bind(
      new Date().toISOString(),
      str(body.install_id),
      str(body.session_id),
      str(body.event) || "run",
      str(body.interface),
      str(body.app_version),
      num(body.schema_version),
      json,
    )
    .run();
}

export default {
  async fetch(request, env, ctx) {
    if (request.method !== "POST") {
      return new Response("Method Not Allowed", { status: 405 });
    }

    const length = Number(request.headers.get("content-length") ?? 0);
    if (!Number.isFinite(length) || length > MAX_BODY_BYTES) {
      return new Response("Payload Too Large", { status: 413 });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return new Response("Bad Request", { status: 400 });
    }

    // Reject anything that is not a beacon shape we know how to store.
    if (!body || typeof body !== "object" || !KNOWN_SCHEMAS.has(body.schema_version)) {
      return new Response("Unsupported schema", { status: 400 });
    }

    const json = JSON.stringify(sanitize(body));
    try {
      writeAnalytics(env, body, json);
    } catch (error) {
      console.error("analytics_write_failed", String(error));
    }
    if (env.DB) {
      // After the response: the client is fire-and-forget, and D1 being slow must not hold it.
      ctx.waitUntil(
        insertEvent(env, body, json).catch((error) =>
          console.error("d1_insert_failed", String(error)),
        ),
      );
    }

    // No body: the client is fire-and-forget and ignores the response.
    return new Response(null, { status: 204 });
  },
};
