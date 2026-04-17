/**
 * HACK MALENDAU 2026 – Predictive Maintenance Simulation Server
 * =============================================================
 * Endpoints:
 *   GET  /stream/:machine_id          Server-Sent Events, 1 reading/sec
 *   GET  /history/:machine_id         Last 7 days of readings (JSON)
 *   POST /alert                       Raise a maintenance alert
 *   POST /schedule-maintenance        (Bonus) Book a maintenance slot
 *   GET  /                            Web dashboard (browser UI)
 *   GET  /alerts                      All alerts raised so far
 */

const express = require("express");
const cors    = require("cors");
const path    = require("path");
const { generateAllHistory, generateReading, MACHINES, BASELINES } = require("./generate-history");

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// ── Precompute history once at startup ────────────────────────────────────────
console.log("⚙️  Generating 7-day sensor history (this takes ~2 seconds)…");
const HISTORY = generateAllHistory();
console.log(`✅  History ready: ${Object.keys(HISTORY).length} machines × ${HISTORY[MACHINES[0]].length} readings each.`);

// ── In-memory stores ──────────────────────────────────────────────────────────
const alerts       = [];  // { id, machine_id, reason, triggered_at, reading }
const maintenances = [];  // { id, machine_id, slot, booked_at }

// ── Live state per machine (drifts over time for interesting SSE streams) ─────
const liveState = {};
for (const m of MACHINES) {
  const b = BASELINES[m];
  liveState[m] = {
    temp:    b.temp,
    vib:     b.vib,
    rpm:     b.rpm,
    current: b.current,
    tick:    0,
  };
}

function rand(min, max) { return Math.random() * (max - min) + min; }
function fix(n, d = 2)  { return parseFloat(n.toFixed(d)); }

/**
 * Advance the live state for one machine by one second tick.
 * Each machine has a unique drift personality so the live stream is interesting.
 */
function nextLiveReading(machineId) {
  const s = liveState[machineId];
  const b = BASELINES[machineId];
  s.tick++;

  let status = "running";
  let temp    = s.temp;
  let vib     = s.vib;
  let rpm     = s.rpm;
  let current = s.current;

  // Slow mean-reversion + noise
  temp    += (b.temp    - temp)    * 0.02 + rand(-0.4, 0.4);
  vib     += (b.vib     - vib)     * 0.02 + rand(-0.05, 0.05);
  rpm     += (b.rpm     - rpm)     * 0.02 + rand(-8,    8);
  current += (b.current - current) * 0.02 + rand(-0.15, 0.15);

  // Machine-specific live anomalies
  if (machineId === "CNC_01") {
    // Steady bearing wear: vib+temp ramp every 5 min
    const rampFactor = Math.min(s.tick / (5 * 60), 1);
    vib     += rampFactor * 0.004;
    temp    += rampFactor * 0.01;
    current += rampFactor * 0.005;
    if (vib > 3.5) status = "warning";
    if (vib > 5.5) status = "fault";
  }

  if (machineId === "CNC_02") {
    // Periodic thermal spike every 3 minutes
    if (s.tick % 180 < 20) {
      temp    += rand(5, 18);
      current += rand(1, 4);
    }
    if (temp > 95)  status = "warning";
    if (temp > 110) status = "fault";
  }

  if (machineId === "PUMP_03") {
    // Random cavitation bursts
    if (Math.random() < 0.04) {
      vib     += rand(1.5, 5);
      current += rand(0.5, 2);
    }
    // Very slow RPM decline simulating clog
    rpm -= 0.02;
    if (vib > 5 || rpm < 2800) status = "warning";
  }

  if (machineId === "CONVEYOR_04") {
    // Mostly healthy — gentle random walk
    if (Math.random() < 0.005) {
      vib += rand(0.5, 1.5);
      status = "warning";
    }
  }

  // Clamp
  temp    = Math.max(20,  Math.min(130, temp));
  vib     = Math.max(0.1, Math.min(12,  vib));
  rpm     = Math.max(100, Math.min(4000, rpm));
  current = Math.max(1,   Math.min(30,  current));

  s.temp    = temp;
  s.vib     = vib;
  s.rpm     = rpm;
  s.current = current;

  return {
    machine_id:     machineId,
    timestamp:      new Date().toISOString(),
    temperature_C:  fix(temp),
    vibration_mm_s: fix(vib),
    rpm:            fix(rpm, 0),
    current_A:      fix(current),
    status,
  };
}

// ── Routes ────────────────────────────────────────────────────────────────────

/**
 * GET /stream/:machine_id
 * Server-Sent Events – pushes one JSON reading per second.
 * Usage: curl http://localhost:3000/stream/CNC_01
 */
app.get("/stream/:machine_id", (req, res) => {
  const { machine_id } = req.params;
  if (!MACHINES.includes(machine_id)) {
    return res.status(404).json({ error: `Unknown machine: ${machine_id}. Valid IDs: ${MACHINES.join(", ")}` });
  }

  res.setHeader("Content-Type",  "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection",    "keep-alive");
  res.flushHeaders();

  // Send a comment to confirm connection
  res.write(`: connected to ${machine_id} stream\n\n`);

  const interval = setInterval(() => {
    const reading = nextLiveReading(machine_id);
    res.write(`data: ${JSON.stringify(reading)}\n\n`);
  }, 1000);

  req.on("close", () => clearInterval(interval));
});

/**
 * GET /history/:machine_id
 * Returns array of ~10 080 readings (1/min × 7 days).
 */
app.get("/history/:machine_id", (req, res) => {
  const { machine_id } = req.params;
  if (!MACHINES.includes(machine_id)) {
    return res.status(404).json({ error: `Unknown machine: ${machine_id}. Valid IDs: ${MACHINES.join(", ")}` });
  }
  res.json({
    machine_id,
    count:    HISTORY[machine_id].length,
    readings: HISTORY[machine_id],
  });
});

/**
 * POST /alert
 * Body: { machine_id, reason, reading? }
 * Stores alert and returns it with a generated ID.
 */
app.post("/alert", (req, res) => {
  const { machine_id, reason, reading } = req.body;

  if (!machine_id || !reason) {
    return res.status(400).json({ error: "Body must include machine_id and reason." });
  }
  if (!MACHINES.includes(machine_id)) {
    return res.status(400).json({ error: `Unknown machine: ${machine_id}. Valid IDs: ${MACHINES.join(", ")}` });
  }

  const alert = {
    id:           `ALERT-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    machine_id,
    reason,
    reading:      reading || null,
    triggered_at: new Date().toISOString(),
  };

  alerts.push(alert);
  console.log(`🚨  [ALERT] ${alert.id} | ${machine_id} | ${reason}`);

  res.status(201).json({ success: true, alert });
});

/**
 * POST /schedule-maintenance  (Bonus)
 * Body: { machine_id, proposed_slot? }
 * Proposes the next available slot (or accepts one) and books it.
 */
app.post("/schedule-maintenance", (req, res) => {
  const { machine_id, proposed_slot } = req.body;

  if (!machine_id) {
    return res.status(400).json({ error: "Body must include machine_id." });
  }
  if (!MACHINES.includes(machine_id)) {
    return res.status(400).json({ error: `Unknown machine: ${machine_id}. Valid IDs: ${MACHINES.join(", ")}` });
  }

  // Auto-assign next available slot (next business morning at 06:00)
  const slot = proposed_slot || (() => {
    const d = new Date();
    d.setDate(d.getDate() + 1);
    d.setHours(6, 0, 0, 0);
    // Skip weekends
    if (d.getDay() === 0) d.setDate(d.getDate() + 1);
    if (d.getDay() === 6) d.setDate(d.getDate() + 2);
    return d.toISOString();
  })();

  const booking = {
    id:         `MAINT-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    machine_id,
    slot,
    booked_at:  new Date().toISOString(),
    technician: "Auto-assigned",
  };

  maintenances.push(booking);
  console.log(`🔧  [MAINTENANCE] ${booking.id} | ${machine_id} | slot: ${slot}`);

  res.status(201).json({ success: true, booking });
});

/**
 * GET /alerts – return all raised alerts (useful for dashboards / agents)
 */
app.get("/alerts", (req, res) => {
  res.json({ count: alerts.length, alerts });
});

/**
 * GET /machines – list all machine IDs and their baselines
 */
app.get("/machines", (req, res) => {
  res.json({ machines: MACHINES, baselines: BASELINES });
});

/**
 * GET / – serve the web dashboard
 */
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log("\n╔══════════════════════════════════════════════════════════╗");
  console.log("║  🏭  HACK MALENDAU 2026 — Sensor Simulation Server       ║");
  console.log("╠══════════════════════════════════════════════════════════╣");
  console.log(`║  Dashboard   →  http://localhost:${PORT}                   ║`);
  console.log(`║  Stream      →  GET  /stream/{machine_id}                ║`);
  console.log(`║  History     →  GET  /history/{machine_id}               ║`);
  console.log(`║  Alert       →  POST /alert                              ║`);
  console.log(`║  Maintenance →  POST /schedule-maintenance               ║`);
  console.log("╠══════════════════════════════════════════════════════════╣");
  console.log(`║  Machines: ${MACHINES.join("  ")}              ║`);
  console.log("╚══════════════════════════════════════════════════════════╝\n");
});
