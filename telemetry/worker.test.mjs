// Tests for the collector, no dependencies: `uv run poe test-worker`.
import assert from "node:assert/strict";
import { test } from "node:test";

import worker from "./worker.js";

// A stand-in for the Worker runtime: records what the collector writes and waits for it.
function fakeEnv({ withDb = true } = {}) {
  const points = [];
  const rows = [];
  const env = { TELEMETRY: { writeDataPoint: (point) => points.push(point) } };
  if (withDb) {
    env.DB = {
      prepare: (sql) => ({
        bind: (...values) => ({
          run: async () => {
            rows.push({ sql, values });
          },
        }),
      }),
    };
  }
  const pending = [];
  const ctx = { waitUntil: (promise) => pending.push(promise) };
  return { env, ctx, points, rows, settle: () => Promise.all(pending) };
}

function beacon(body, init = {}) {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return new Request("https://telemetry.tagger.photo", {
    method: "POST",
    body: text,
    headers: { "content-type": "application/json", "content-length": String(text.length) },
    ...init,
  });
}

const RUN = {
  schema_version: 3,
  event: "run",
  install_id: "7d0f3c4e-9d52-4b8a-8a1c-2a3e4f5a6b7c",
  session_id: "s-1",
  app_version: "0.10.0",
  interface: "gui",
  provider: "lmstudio",
  batch_size: 12,
  dry_run: false,
  settings: { sidecar_mode: "raw", backup: true, jpeg_dimensions: 640 },
  features: { viewer: 3 },
};

test("a v3 beacon lands in Analytics Engine and in D1 with the same JSON", async () => {
  const { env, ctx, points, rows, settle } = fakeEnv();
  const response = await worker.fetch(beacon(RUN), env, ctx);
  await settle();

  assert.equal(response.status, 204);
  assert.equal(points.length, 1);
  const [point] = points;
  assert.deepEqual(point.indexes, [RUN.install_id]);
  assert.equal(point.blobs[1], "gui");
  assert.equal(point.blobs[11], "run");
  assert.equal(point.doubles[1], 12);
  assert.equal(point.doubles[12], 0, "a boolean dry_run still lands as 0 or 1");
  const stored = JSON.parse(point.blobs[18]);
  assert.deepEqual(stored.settings, RUN.settings);

  assert.equal(rows.length, 1);
  const [, installId, sessionId, event, iface, version, schema, payload] = rows[0].values;
  assert.deepEqual(
    [installId, sessionId, event, iface, version, schema],
    [RUN.install_id, "s-1", "run", "gui", "0.10.0", 3],
  );
  assert.equal(payload, point.blobs[18]);
});

test("a true boolean lands as 1, not 0", async () => {
  const { env, ctx, points, settle } = fakeEnv();
  await worker.fetch(beacon({ ...RUN, dry_run: true }), env, ctx);
  await settle();

  assert.equal(points[0].doubles[12], 1);
});

test("the collector still works without a D1 binding", async () => {
  const { env, ctx, points, settle } = fakeEnv({ withDb: false });
  const response = await worker.fetch(beacon(RUN), env, ctx);
  await settle();
  assert.equal(response.status, 204);
  assert.equal(points.length, 1);
});

test("v1 and v2 beacons from older installs are still accepted", async () => {
  for (const schema_version of [1, 2]) {
    const { env, ctx, points } = fakeEnv();
    const response = await worker.fetch(beacon({ schema_version, install_id: "x" }), env, ctx);
    assert.equal(response.status, 204);
    assert.equal(points[0].blobs[11], "run", "a beacon without an event is a run");
  }
});

test("sanitizing drops odd keys, lists, deep nesting, and clamps long strings", async () => {
  const { env, ctx, points } = fakeEnv();
  await worker.fetch(
    beacon({
      schema_version: 3,
      "Bad Key": 1,
      list: [1, 2],
      nested: { deeper: { deepest: 1 }, fine: "ok", crash_stack: "y".repeat(5000) },
      long: "x".repeat(5000),
      crash_stack: "z".repeat(9000),
    }),
    env,
    ctx,
  );
  const stored = JSON.parse(points[0].blobs[18]);
  assert.equal("Bad Key" in stored, false);
  assert.equal("list" in stored, false);
  assert.deepEqual(Object.keys(stored.nested), ["fine", "crash_stack"]);
  assert.equal(stored.long.length, 1000);
  // Only a top-level crash stack gets the longer clamp.
  assert.equal(stored.crash_stack.length, 6000);
  assert.equal(stored.nested.crash_stack.length, 1000);
});

test("unknown schemas, bad JSON, big bodies, and other methods are refused", async () => {
  const { env, ctx } = fakeEnv();
  assert.equal((await worker.fetch(beacon({ schema_version: 99 }), env, ctx)).status, 400);
  assert.equal((await worker.fetch(beacon("{not json"), env, ctx)).status, 400);
  const huge = beacon({ schema_version: 3, pad: "x".repeat(40000) });
  assert.equal((await worker.fetch(huge, env, ctx)).status, 413);
  const get = new Request("https://telemetry.tagger.photo", { method: "GET" });
  assert.equal((await worker.fetch(get, env, ctx)).status, 405);
});

test("a failing D1 insert never turns into an error response", async () => {
  const { env, ctx, settle } = fakeEnv();
  env.DB.prepare = () => ({
    bind: () => ({
      run: async () => {
        throw new Error("D1 is down");
      },
    }),
  });
  const response = await worker.fetch(beacon(RUN), env, ctx);
  await settle();
  assert.equal(response.status, 204);
});
