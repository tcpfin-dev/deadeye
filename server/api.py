#!/usr/bin/env python3
"""
Unified API Server — Multi-Agent Anomaly Detection Dashboard Backend

Inference agents must register on startup to receive a server-assigned node_id:
  • POST /api/register              — register a new node, returns { node_id }

Receives streaming inference data from registered agents:
  • Maintenance Agent  → POST /api/maintenance/reading
  • Power Agent        → POST /api/power/reading

Heartbeat / liveness:
  • POST /api/heartbeat             — explicit keep-alive ping from agents
  • Automatic: each reading ingestion also counts as a heartbeat
  • Nodes with no activity for HEARTBEAT_TIMEOUT_S seconds are marked DEAD
  • DELETE /api/nodes/<agent_type>/<node_id>  — remove a dead/unwanted node

Agent tool endpoints (LLM agents call these via LangChain/LangGraph):
  • GET  /api/agent/system_status                   — high-level system snapshot
  • GET  /api/agent/node_detail/<type>/<node_id>    — deep-dive single node
  • GET  /api/agent/failures                        — investigate failure events
  • GET  /api/agent/sensor_summary/<type>           — sensor value statistics
  • GET  /api/agent/compare_nodes/<type>            — side-by-side node comparison
  • GET  /api/agent/correlation                     — cross-agent event correlation
  • GET  /api/agent/recent_activity                 — unified chronological feed
  • GET  /api/agent/search_readings                 — filter readings by sensor thresholds

Unregistered node_ids are rejected (403).

Exposes dashboard-ready endpoints:
  • GET  /api/maintenance/latest       — last N readings (all nodes or ?node_id=X)
  • GET  /api/power/latest             — last N readings (all nodes or ?node_id=X)
  • GET  /api/maintenance/stats        — live accuracy, confusion matrix (all nodes or ?node_id=X)
  • GET  /api/power/stats              — live accuracy, confusion matrix (all nodes or ?node_id=X)
  • GET  /api/maintenance/alerts       — recent failure alerts only
  • GET  /api/power/alerts             — recent attack alerts only
  • GET  /api/nodes                    — list all registered nodes with metadata + liveness
  • GET  /api/overview                 — combined health summary across all agents
  • GET  /api/health                   — server heartbeat

Usage:
    python api.py
    python api.py --port 5000
    python api.py --host 0.0.0.0 --port 8080
    python api.py --timeout 60       # heartbeat timeout in seconds (default: 30)
"""

import argparse
import statistics
import threading
import time as _time
from datetime import datetime
from collections import deque

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# In-memory stores (thread-safe via lock)
# ─────────────────────────────────────────────

MAX_READINGS = 5000
MAX_ALERTS   = 500
HEARTBEAT_TIMEOUT_S = 30   # seconds — overridden by CLI --timeout

lock = threading.Lock()

# Auto-increment counter for server-assigned node IDs
_node_counter = {"maintenance": 0, "power": 0}

# Registry of valid node IDs — only registered nodes may ingest data
# Keyed by agent_type -> node_id -> metadata dict
_registered_nodes = {"maintenance": {}, "power": {}}


def make_node_store():
    """Create a fresh per-node data store."""
    now = _time.time()
    return {
        "readings": deque(maxlen=MAX_READINGS),
        "alerts":   deque(maxlen=MAX_ALERTS),
        "stats": {
            "total": 0, "correct": 0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "alerts_fired": 0,
            "last_updated": None,
        },
        "created_at": datetime.now().isoformat(),
        "last_heartbeat": now,       # monotonic seconds
        "last_heartbeat_iso": datetime.now().isoformat(),
    }

# Keyed by agent_type -> node_id -> store
# e.g. nodes["maintenance"]["node_1"] = { readings, alerts, stats }
nodes = {
    "maintenance": {},
    "power": {},
}


def touch_heartbeat(store):
    """Update the heartbeat timestamp on a node store. Must be called within lock."""
    store["last_heartbeat"] = _time.time()
    store["last_heartbeat_iso"] = datetime.now().isoformat()


def node_liveness(store):
    """Return 'alive', 'idle', or 'dead' based on heartbeat age."""
    if store["stats"]["total"] == 0 and (_time.time() - store["last_heartbeat"]) < HEARTBEAT_TIMEOUT_S:
        return "idle"
    elapsed = _time.time() - store["last_heartbeat"]
    if elapsed > HEARTBEAT_TIMEOUT_S:
        return "dead"
    return "alive"


def get_or_create_node(agent_type, node_id):
    """Get or lazily create the data store for an already-registered node.
    Must be called within lock. Returns None if node_id is not registered."""
    if node_id not in _registered_nodes[agent_type]:
        return None
    if node_id not in nodes[agent_type]:
        nodes[agent_type][node_id] = make_node_store()
    return nodes[agent_type][node_id]


def compute_metrics(s):
    """Derive accuracy, precision, recall, f1 from running confusion counts."""
    total = s["total"]
    tp, fp, tn, fn = s["tp"], s["fp"], s["tn"], s["fn"]
    acc  = s["correct"] / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {
        "total_readings": total,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "alerts_fired": s["alerts_fired"],
        "last_updated": s["last_updated"],
    }


def aggregate_stats(agent_type):
    """Aggregate stats across all nodes of an agent type."""
    agg = {"total": 0, "correct": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
           "alerts_fired": 0, "last_updated": None}
    for nid, store in nodes[agent_type].items():
        s = store["stats"]
        for k in ["total", "correct", "tp", "fp", "tn", "fn", "alerts_fired"]:
            agg[k] += s[k]
        if s["last_updated"]:
            if agg["last_updated"] is None or s["last_updated"] > agg["last_updated"]:
                agg["last_updated"] = s["last_updated"]
    return agg


# ═════════════════════════════════════════════
#  NODE REGISTRATION  (called by inference agents on startup)
# ═════════════════════════════════════════════

@app.route("/api/register", methods=["POST"])
def register_node():
    """Register a new inference node.

    Request JSON:
        agent_type  (str, required): "maintenance" or "power"
        node_name   (str, optional): human-readable label (e.g. "turbine-A3")

    Response 201:
        node_id     (str): server-assigned unique ID, e.g. "node_3"
        agent_type  (str): echo back
        node_name   (str): the label (defaults to the node_id)
        registered_at (str): ISO timestamp
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty payload"}), 400

    agent_type = data.get("agent_type")
    if agent_type not in ("maintenance", "power"):
        return jsonify({"error": "agent_type must be 'maintenance' or 'power'"}), 400

    node_name = data.get("node_name")  # optional friendly label

    with lock:
        _node_counter[agent_type] += 1
        node_id = f"node_{_node_counter[agent_type]}"
        registered_at = datetime.now().isoformat()

        _registered_nodes[agent_type][node_id] = {
            "node_name": node_name or node_id,
            "registered_at": registered_at,
        }

        # Pre-create the data store so it shows up immediately in /api/nodes
        nodes[agent_type][node_id] = make_node_store()

    return jsonify({
        "status": "ok",
        "node_id": node_id,
        "agent_type": agent_type,
        "node_name": node_name or node_id,
        "registered_at": registered_at,
    }), 201


# ═════════════════════════════════════════════
#  HEARTBEAT  (explicit keep-alive from agents)
# ═════════════════════════════════════════════

@app.route("/api/heartbeat", methods=["POST"])
def heartbeat_ping():
    """Explicit heartbeat from an inference agent.

    Request JSON:
        agent_type (str): "maintenance" or "power"
        node_id    (str): the server-assigned node_id

    Response 200:
        status     (str): "ok"
        liveness   (str): current liveness after the ping ("alive")
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty payload"}), 400

    agent_type = data.get("agent_type")
    node_id = data.get("node_id")

    if agent_type not in ("maintenance", "power"):
        return jsonify({"error": "agent_type must be 'maintenance' or 'power'"}), 400
    if not node_id:
        return jsonify({"error": "node_id is required"}), 400

    with lock:
        store = get_or_create_node(agent_type, node_id)
        if store is None:
            return jsonify({"error": f"node_id '{node_id}' not registered"}), 403
        touch_heartbeat(store)
        liveness = node_liveness(store)

    return jsonify({"status": "ok", "node_id": node_id, "liveness": liveness}), 200


# ═════════════════════════════════════════════
#  INGEST ENDPOINTS  (called by inference agents)
# ═════════════════════════════════════════════

def ingest_reading(agent_type, alert_status_key):
    """Generic ingest handler for both maintenance and power agents."""
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty payload"}), 400

    node_id = data.get("node_id")
    if not node_id:
        return jsonify({"error": "node_id is required. Register first via POST /api/register"}), 400

    data["received_at"] = datetime.now().isoformat()

    pred   = data.get("prediction", 0)
    actual = data.get("actual", 0)

    with lock:
        store = get_or_create_node(agent_type, node_id)
        if store is None:
            return jsonify({
                "error": f"node_id '{node_id}' is not registered for agent '{agent_type}'. "
                         f"Register first via POST /api/register"
            }), 403

        # Each reading counts as a heartbeat
        touch_heartbeat(store)

        store["readings"].append(data)

        s = store["stats"]
        s["total"] += 1
        if pred == actual:
            s["correct"] += 1
        if pred == 1 and actual == 1:   s["tp"] += 1
        elif pred == 1 and actual == 0: s["fp"] += 1
        elif pred == 0 and actual == 0: s["tn"] += 1
        else:                           s["fn"] += 1

        if data.get("status") == alert_status_key:
            s["alerts_fired"] += 1
            store["alerts"].append(data)

        s["last_updated"] = data["received_at"]
        total = s["total"]

    return jsonify({
        "status": "ok",
        "agent": agent_type,
        "node_id": node_id,
        "reading_count": total,
    }), 201


@app.route("/api/maintenance/reading", methods=["POST"])
def ingest_maintenance():
    return ingest_reading("maintenance", "FAILURE")


@app.route("/api/power/reading", methods=["POST"])
def ingest_power():
    return ingest_reading("power", "ATTACK")


# ═════════════════════════════════════════════
#  QUERY ENDPOINTS  (called by dashboard)
# ═════════════════════════════════════════════

@app.route("/api/maintenance/latest", methods=["GET"])
def get_maintenance_latest():
    return _get_latest("maintenance")

@app.route("/api/power/latest", methods=["GET"])
def get_power_latest():
    return _get_latest("power")

def _get_latest(agent_type):
    n = min(int(request.args.get("n", 50)), MAX_READINGS)
    node_id = request.args.get("node_id", None)
    with lock:
        if node_id:
            store = nodes[agent_type].get(node_id)
            readings = list(store["readings"])[-n:] if store else []
        else:
            all_r = []
            for nid, store in nodes[agent_type].items():
                all_r.extend(store["readings"])
            all_r.sort(key=lambda r: r.get("received_at", ""))
            readings = all_r[-n:]
    return jsonify({"agent": agent_type, "node_id": node_id, "count": len(readings), "readings": readings})


@app.route("/api/maintenance/stats", methods=["GET"])
def get_maintenance_stats():
    return _get_stats("maintenance")

@app.route("/api/power/stats", methods=["GET"])
def get_power_stats():
    return _get_stats("power")

def _get_stats(agent_type):
    node_id = request.args.get("node_id", None)
    with lock:
        if node_id:
            store = nodes[agent_type].get(node_id)
            metrics = compute_metrics(store["stats"]) if store else compute_metrics(
                {"total":0,"correct":0,"tp":0,"fp":0,"tn":0,"fn":0,"alerts_fired":0,"last_updated":None})
        else:
            metrics = compute_metrics(aggregate_stats(agent_type))
    return jsonify({"agent": agent_type, "node_id": node_id, **metrics})


@app.route("/api/maintenance/alerts", methods=["GET"])
def get_maintenance_alerts():
    return _get_alerts("maintenance")

@app.route("/api/power/alerts", methods=["GET"])
def get_power_alerts():
    return _get_alerts("power")

def _get_alerts(agent_type):
    n = min(int(request.args.get("n", 50)), MAX_ALERTS)
    node_id = request.args.get("node_id", None)
    with lock:
        if node_id:
            store = nodes[agent_type].get(node_id)
            alerts = list(store["alerts"])[-n:] if store else []
        else:
            all_a = []
            for nid, store in nodes[agent_type].items():
                all_a.extend(store["alerts"])
            all_a.sort(key=lambda a: a.get("received_at", ""))
            alerts = all_a[-n:]
    return jsonify({"agent": agent_type, "node_id": node_id, "count": len(alerts), "alerts": alerts})


# ═════════════════════════════════════════════
#  NODE REGISTRY + DELETION
# ═════════════════════════════════════════════

@app.route("/api/nodes", methods=["GET"])
def get_nodes():
    """List all registered nodes with their metadata, stats, and liveness."""
    result = {"maintenance": [], "power": []}
    with lock:
        for agent_type in ["maintenance", "power"]:
            for nid, store in nodes[agent_type].items():
                metrics = compute_metrics(store["stats"])
                last_reading = None
                if store["readings"]:
                    last_reading = dict(store["readings"][-1])
                reg = _registered_nodes.get(agent_type, {}).get(nid, {})
                liveness = node_liveness(store)
                elapsed = _time.time() - store["last_heartbeat"]
                result[agent_type].append({
                    "node_id": nid,
                    "node_name": reg.get("node_name", nid),
                    "registered_at": reg.get("registered_at", store["created_at"]),
                    "created_at": store["created_at"],
                    "stats": metrics,
                    "last_reading": last_reading,
                    "liveness": liveness,
                    "last_heartbeat": store["last_heartbeat_iso"],
                    "heartbeat_age_s": round(elapsed, 1),
                })
    return jsonify(result)


@app.route("/api/nodes/<agent_type>/<node_id>", methods=["DELETE"])
def delete_node(agent_type, node_id):
    """Remove a node from the registry and purge its data.

    Typically used from the dashboard to clean up dead nodes.
    """
    if agent_type not in ("maintenance", "power"):
        return jsonify({"error": "agent_type must be 'maintenance' or 'power'"}), 400

    with lock:
        if node_id not in _registered_nodes.get(agent_type, {}):
            return jsonify({"error": f"Node '{node_id}' not found in '{agent_type}'"}), 404

        _registered_nodes[agent_type].pop(node_id, None)
        nodes[agent_type].pop(node_id, None)

    return jsonify({
        "status": "ok",
        "message": f"Node '{node_id}' removed from '{agent_type}'",
    }), 200


# ═════════════════════════════════════════════
#  AGENT TOOL ENDPOINTS
#  (LLM agents call these as tools via LangChain/LangGraph
#   to investigate stats, failures, anomalies, and correlations)
# ═════════════════════════════════════════════

@app.route("/api/agent/system_status", methods=["GET"])
def agent_system_status():
    """High-level system snapshot for an LLM agent to orient itself.

    Returns a concise summary: which agent types are running, how many nodes
    are alive/dead, total readings, alert counts, current accuracy per agent,
    and whether any node is in a degraded state.  This is the first call an
    agent should make to understand the current situation.

    Response:
        system_status    (str): "IDLE" | "HEALTHY" | "ALERT" | "DEGRADED"
        server_time      (str): ISO timestamp
        agents           (dict): per-agent summary with accuracy, nodes, alerts
        degraded_nodes   (list): nodes with accuracy < 80% or liveness != alive
    """
    with lock:
        result = {"maintenance": {}, "power": {}}
        degraded = []

        for agent_type in ["maintenance", "power"]:
            agg = aggregate_stats(agent_type)
            metrics = compute_metrics(agg)
            node_count = len(nodes[agent_type])
            alive = 0
            dead = 0

            for nid, store in nodes[agent_type].items():
                liveness = node_liveness(store)
                if liveness == "dead":
                    dead += 1
                else:
                    alive += 1

                # Flag degraded nodes
                nm = compute_metrics(store["stats"])
                is_degraded = False
                reasons = []
                if liveness == "dead":
                    is_degraded = True
                    reasons.append("node is dead (heartbeat timeout)")
                if nm["total_readings"] >= 20 and nm["accuracy"] < 0.80:
                    is_degraded = True
                    reasons.append(f"low accuracy ({nm['accuracy']:.1%})")
                if nm["total_readings"] >= 20 and nm["precision"] > 0 and nm["precision"] < 0.50:
                    is_degraded = True
                    reasons.append(f"high false-positive rate (precision {nm['precision']:.1%})")
                if nm["total_readings"] >= 20 and nm["recall"] < 0.50:
                    is_degraded = True
                    reasons.append(f"missing detections (recall {nm['recall']:.1%})")

                if is_degraded:
                    reg = _registered_nodes.get(agent_type, {}).get(nid, {})
                    degraded.append({
                        "node_id": nid,
                        "agent_type": agent_type,
                        "node_name": reg.get("node_name", nid),
                        "liveness": liveness,
                        "accuracy": nm["accuracy"],
                        "precision": nm["precision"],
                        "recall": nm["recall"],
                        "total_readings": nm["total_readings"],
                        "reasons": reasons,
                    })

            result[agent_type] = {
                "node_count": node_count,
                "alive": alive,
                "dead": dead,
                "total_readings": metrics["total_readings"],
                "alerts_fired": metrics["alerts_fired"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }

    total_readings = result["maintenance"]["total_readings"] + result["power"]["total_readings"]
    total_alerts = result["maintenance"]["alerts_fired"] + result["power"]["alerts_fired"]

    if total_readings == 0:
        status = "IDLE"
    elif len(degraded) > 0:
        status = "DEGRADED"
    elif total_alerts > 0:
        status = "ALERT"
    else:
        status = "HEALTHY"

    return jsonify({
        "system_status": status,
        "server_time": datetime.now().isoformat(),
        "total_readings": total_readings,
        "total_alerts": total_alerts,
        "agents": result,
        "degraded_nodes": degraded,
    })


@app.route("/api/agent/node_detail/<agent_type>/<node_id>", methods=["GET"])
def agent_node_detail(agent_type, node_id):
    """Deep-dive into a single node's performance and recent activity.

    Returns full metrics, the last N readings, recent alerts, and computed
    trend indicators (is accuracy improving or degrading over the last window).
    An LLM agent uses this after seeing a degraded node in /system_status.

    Path params:
        agent_type  (str): "maintenance" or "power"
        node_id     (str): e.g. "node_1"

    Query params:
        n           (int): number of recent readings to return (default 20, max 100)

    Response:
        node_id, agent_type, node_name, liveness, registered_at,
        metrics (full confusion matrix + derived),
        recent_readings (last N with all sensor fields),
        recent_alerts (last 10),
        trend (accuracy/alert_rate over recent window vs overall)
    """
    if agent_type not in ("maintenance", "power"):
        return jsonify({"error": "agent_type must be 'maintenance' or 'power'"}), 400

    n = min(int(request.args.get("n", 20)), 100)

    with lock:
        store = nodes[agent_type].get(node_id)
        if store is None:
            return jsonify({"error": f"Node '{node_id}' not found in '{agent_type}'"}), 404

        reg = _registered_nodes.get(agent_type, {}).get(node_id, {})
        metrics = compute_metrics(store["stats"])
        liveness = node_liveness(store)
        recent_readings = list(store["readings"])[-n:]
        recent_alerts = list(store["alerts"])[-10:]
        elapsed = _time.time() - store["last_heartbeat"]

        # ── Trend analysis: compare last 50 readings vs overall ──
        trend = {}
        all_readings = list(store["readings"])
        window = all_readings[-50:] if len(all_readings) >= 50 else all_readings
        if len(window) >= 10:
            w_correct = sum(1 for r in window if r.get("prediction") == r.get("actual"))
            w_alerts = sum(1 for r in window if r.get("status") in ("FAILURE", "ATTACK"))
            w_acc = w_correct / len(window)
            w_alert_rate = w_alerts / len(window)
            overall_acc = metrics["accuracy"]
            trend = {
                "window_size": len(window),
                "window_accuracy": round(w_acc, 4),
                "overall_accuracy": overall_acc,
                "accuracy_delta": round(w_acc - overall_acc, 4),
                "window_alert_rate": round(w_alert_rate, 4),
                "trending": "improving" if w_acc > overall_acc + 0.02
                            else "degrading" if w_acc < overall_acc - 0.02
                            else "stable",
            }

    return jsonify({
        "node_id": node_id,
        "agent_type": agent_type,
        "node_name": reg.get("node_name", node_id),
        "liveness": liveness,
        "registered_at": reg.get("registered_at"),
        "heartbeat_age_s": round(elapsed, 1),
        "metrics": metrics,
        "trend": trend,
        "recent_readings": recent_readings,
        "recent_alerts": recent_alerts,
    })


@app.route("/api/agent/failures", methods=["GET"])
def agent_failures():
    """Retrieve detailed failure/attack events with full sensor context.

    An LLM agent calls this to investigate what went wrong: which readings
    triggered alerts, what were the sensor values at that moment, and whether
    the prediction was correct (true positive) or wrong (false negative that
    was actually a failure, or false positive that was a false alarm).

    Query params:
        agent_type  (str, optional): filter to "maintenance" or "power"
        node_id     (str, optional): filter to a specific node
        n           (int):  max results (default 20, max 100)
        kind        (str):  "all" (default) | "true_positive" | "false_positive"
                            | "false_negative" | "missed"
                            ("missed" = fn, alias for false_negative)

    Response:
        failures: list of reading dicts annotated with classification_type
    """
    filter_type = request.args.get("agent_type", None)
    filter_node = request.args.get("node_id", None)
    n = min(int(request.args.get("n", 20)), 100)
    kind = request.args.get("kind", "all").lower()

    if filter_type and filter_type not in ("maintenance", "power"):
        return jsonify({"error": "agent_type must be 'maintenance' or 'power'"}), 400

    # Map kind aliases
    kind_map = {"missed": "false_negative", "false_alarm": "false_positive"}
    kind = kind_map.get(kind, kind)

    results = []
    with lock:
        for agent_type in ["maintenance", "power"]:
            if filter_type and agent_type != filter_type:
                continue
            for nid, store in nodes[agent_type].items():
                if filter_node and nid != filter_node:
                    continue
                for r in store["readings"]:
                    pred = r.get("prediction", 0)
                    actual = r.get("actual", 0)

                    # Classify
                    if pred == 1 and actual == 1:
                        ctype = "true_positive"
                    elif pred == 1 and actual == 0:
                        ctype = "false_positive"
                    elif pred == 0 and actual == 1:
                        ctype = "false_negative"
                    else:
                        continue  # true negatives are not interesting here

                    if kind != "all" and ctype != kind:
                        continue

                    entry = dict(r)
                    entry["classification"] = ctype
                    entry["agent_type"] = agent_type
                    entry["node_id_source"] = nid
                    results.append(entry)

    # Sort by time descending, take last n
    results.sort(key=lambda x: x.get("received_at", x.get("timestamp", "")), reverse=True)
    results = results[:n]

    return jsonify({
        "count": len(results),
        "kind_filter": kind,
        "failures": results,
    })


@app.route("/api/agent/sensor_summary/<agent_type>", methods=["GET"])
def agent_sensor_summary(agent_type):
    """Statistical summary of sensor values — min, max, mean, std for each
    numeric field across recent readings.  Helps an LLM agent understand
    normal operating ranges and spot outliers.

    Path params:
        agent_type  (str): "maintenance" or "power"

    Query params:
        node_id     (str, optional): specific node or all
        n           (int): window of recent readings to summarise (default 100, max 500)

    Response:
        sensors: { field_name: { min, max, mean, std, count } }
        alert_sensor_profile: same stats but only for readings where status was FAILURE/ATTACK
    """
    if agent_type not in ("maintenance", "power"):
        return jsonify({"error": "agent_type must be 'maintenance' or 'power'"}), 400

    filter_node = request.args.get("node_id", None)
    n = min(int(request.args.get("n", 100)), 500)

    # Known numeric sensor fields per agent type
    sensor_fields = {
        "maintenance": ["tool_wear", "torque", "power_w", "air_temp",
                        "process_temp", "rotational_speed", "probability"],
        "power": ["r1_vh", "r2_vh", "r3_vh", "r4_vh", "relay_status", "probability"],
    }
    fields = sensor_fields.get(agent_type, [])

    with lock:
        all_readings = []
        for nid, store in nodes[agent_type].items():
            if filter_node and nid != filter_node:
                continue
            all_readings.extend(list(store["readings"])[-n:])

    # Sort and trim
    all_readings.sort(key=lambda x: x.get("received_at", x.get("timestamp", "")))
    all_readings = all_readings[-n:]

    def summarise(readings, fields):
        summary = {}
        for f in fields:
            vals = [r[f] for r in readings if f in r and r[f] is not None]
            if not vals:
                continue
            summary[f] = {
                "count": len(vals),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals), 4) if len(vals) >= 2 else 0.0,
            }
        return summary

    alert_key = "FAILURE" if agent_type == "maintenance" else "ATTACK"
    normal_readings = [r for r in all_readings if r.get("status") != alert_key]
    alert_readings = [r for r in all_readings if r.get("status") == alert_key]

    return jsonify({
        "agent_type": agent_type,
        "node_id": filter_node,
        "window_size": len(all_readings),
        "sensors_overall": summarise(all_readings, fields),
        "sensors_normal": summarise(normal_readings, fields),
        "sensors_alert": summarise(alert_readings, fields),
    })


@app.route("/api/agent/compare_nodes/<agent_type>", methods=["GET"])
def agent_compare_nodes(agent_type):
    """Side-by-side performance comparison of all nodes of a given agent type.

    Useful for an LLM agent to quickly spot which node is underperforming
    relative to its peers, or to answer "which node has the most false alarms?"

    Path params:
        agent_type  (str): "maintenance" or "power"

    Response:
        nodes: list of { node_id, node_name, liveness, metrics, alert_rate, readings_count }
        best_node, worst_node (by f1_score)
    """
    if agent_type not in ("maintenance", "power"):
        return jsonify({"error": "agent_type must be 'maintenance' or 'power'"}), 400

    node_list = []
    with lock:
        for nid, store in nodes[agent_type].items():
            reg = _registered_nodes.get(agent_type, {}).get(nid, {})
            metrics = compute_metrics(store["stats"])
            total = metrics["total_readings"]
            alert_rate = metrics["alerts_fired"] / total if total > 0 else 0
            node_list.append({
                "node_id": nid,
                "node_name": reg.get("node_name", nid),
                "liveness": node_liveness(store),
                "total_readings": total,
                "alerts_fired": metrics["alerts_fired"],
                "alert_rate": round(alert_rate, 4),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "confusion_matrix": metrics["confusion_matrix"],
            })

    # Determine best/worst by f1 (only nodes with enough data)
    scored = [n for n in node_list if n["total_readings"] >= 10]
    best = max(scored, key=lambda x: x["f1_score"]) if scored else None
    worst = min(scored, key=lambda x: x["f1_score"]) if scored else None

    return jsonify({
        "agent_type": agent_type,
        "node_count": len(node_list),
        "nodes": node_list,
        "best_node": best["node_id"] if best else None,
        "worst_node": worst["node_id"] if worst else None,
    })


@app.route("/api/agent/correlation", methods=["GET"])
def agent_correlation():
    """Cross-agent temporal correlation — checks whether maintenance failures
    and power attacks tend to co-occur within a time window.

    An LLM agent calls this to investigate whether, say, a power grid attack
    is correlated with a spike in equipment failures.

    Query params:
        window_s    (int): time window in seconds to consider "co-occurring" (default 60)
        n           (int): max number of correlated event pairs to return (default 20)

    Response:
        correlated_events: list of { maintenance_alert, power_alert, time_gap_s }
        summary: { maintenance_alerts_total, power_alerts_total,
                   correlated_count, correlation_pct }
    """
    window_s = int(request.args.get("window_s", 60))
    n = min(int(request.args.get("n", 20)), 100)

    with lock:
        m_alerts = []
        for nid, store in nodes["maintenance"].items():
            for a in store["alerts"]:
                entry = dict(a)
                entry["_node_id"] = nid
                m_alerts.append(entry)

        p_alerts = []
        for nid, store in nodes["power"].items():
            for a in store["alerts"]:
                entry = dict(a)
                entry["_node_id"] = nid
                p_alerts.append(entry)

    def parse_ts(r):
        ts = r.get("received_at") or r.get("timestamp", "")
        try:
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None

    correlated = []
    for ma in m_alerts:
        ma_ts = parse_ts(ma)
        if ma_ts is None:
            continue
        for pa in p_alerts:
            pa_ts = parse_ts(pa)
            if pa_ts is None:
                continue
            gap = abs((ma_ts - pa_ts).total_seconds())
            if gap <= window_s:
                correlated.append({
                    "maintenance_alert": {
                        "node_id": ma["_node_id"],
                        "timestamp": ma.get("received_at") or ma.get("timestamp"),
                        "probability": ma.get("probability"),
                        "status": ma.get("status"),
                    },
                    "power_alert": {
                        "node_id": pa["_node_id"],
                        "timestamp": pa.get("received_at") or pa.get("timestamp"),
                        "probability": pa.get("probability"),
                        "status": pa.get("status"),
                    },
                    "time_gap_s": round(gap, 1),
                })

    correlated.sort(key=lambda x: x["time_gap_s"])
    correlated = correlated[:n]

    total_m = len(m_alerts)
    total_p = len(p_alerts)
    corr_count = len(correlated)
    denom = min(total_m, total_p) if min(total_m, total_p) > 0 else 1

    return jsonify({
        "window_s": window_s,
        "summary": {
            "maintenance_alerts_total": total_m,
            "power_alerts_total": total_p,
            "correlated_count": corr_count,
            "correlation_pct": round(corr_count / denom * 100, 1),
        },
        "correlated_events": correlated,
    })


@app.route("/api/agent/recent_activity", methods=["GET"])
def agent_recent_activity():
    """Unified chronological feed of the most recent events across all agents
    and nodes.  Returns readings, alerts, and state changes in time order.

    An LLM agent uses this to answer "what happened in the last 5 minutes?"
    or "show me the most recent activity".

    Query params:
        n           (int): max events to return (default 30, max 200)
        agent_type  (str, optional): filter to "maintenance" or "power"
        alerts_only (str): "true" to show only alert events (default "false")

    Response:
        events: list of { event_type, agent_type, node_id, timestamp, ... }
    """
    n = min(int(request.args.get("n", 30)), 200)
    filter_type = request.args.get("agent_type", None)
    alerts_only = request.args.get("alerts_only", "false").lower() == "true"

    events = []
    with lock:
        for agent_type in ["maintenance", "power"]:
            if filter_type and agent_type != filter_type:
                continue
            alert_key = "FAILURE" if agent_type == "maintenance" else "ATTACK"
            for nid, store in nodes[agent_type].items():
                readings = list(store["readings"])
                for r in readings:
                    is_alert = r.get("status") == alert_key
                    if alerts_only and not is_alert:
                        continue
                    entry = {
                        "event_type": "alert" if is_alert else "reading",
                        "agent_type": agent_type,
                        "node_id": nid,
                        "timestamp": r.get("received_at") or r.get("timestamp"),
                        "status": r.get("status"),
                        "probability": r.get("probability"),
                        "prediction": r.get("prediction"),
                        "actual": r.get("actual"),
                    }
                    # Include key sensor fields
                    if agent_type == "maintenance":
                        for f in ["tool_wear", "torque", "power_w"]:
                            if f in r:
                                entry[f] = r[f]
                    else:
                        for f in ["r1_vh", "r2_vh", "r3_vh", "r4_vh"]:
                            if f in r:
                                entry[f] = r[f]
                    events.append(entry)

    events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    events = events[:n]

    return jsonify({
        "count": len(events),
        "events": events,
    })


@app.route("/api/agent/search_readings", methods=["GET"])
def agent_search_readings():
    """Search/filter readings by sensor value thresholds.

    An LLM agent calls this to answer questions like "find all readings where
    tool_wear > 200" or "show me readings with probability > 0.9".

    Query params:
        agent_type  (str, required): "maintenance" or "power"
        node_id     (str, optional): filter to specific node
        field       (str, required): sensor field name to filter on
                    (e.g. "tool_wear", "torque", "probability", "r1_vh")
        min_val     (float, optional): minimum value (inclusive)
        max_val     (float, optional): maximum value (inclusive)
        n           (int): max results (default 20, max 100)

    Response:
        readings: list of matching reading dicts
    """
    agent_type = request.args.get("agent_type")
    if agent_type not in ("maintenance", "power"):
        return jsonify({"error": "agent_type is required ('maintenance' or 'power')"}), 400

    field = request.args.get("field")
    if not field:
        return jsonify({"error": "field is required (e.g. 'tool_wear', 'probability')"}), 400

    filter_node = request.args.get("node_id", None)
    n = min(int(request.args.get("n", 20)), 100)
    min_val = request.args.get("min_val", None)
    max_val = request.args.get("max_val", None)
    if min_val is not None:
        min_val = float(min_val)
    if max_val is not None:
        max_val = float(max_val)

    results = []
    with lock:
        for nid, store in nodes[agent_type].items():
            if filter_node and nid != filter_node:
                continue
            for r in store["readings"]:
                val = r.get(field)
                if val is None:
                    continue
                if min_val is not None and val < min_val:
                    continue
                if max_val is not None and val > max_val:
                    continue
                entry = dict(r)
                entry["node_id_source"] = nid
                results.append(entry)

    results.sort(key=lambda x: x.get(field, 0), reverse=True)
    results = results[:n]

    return jsonify({
        "agent_type": agent_type,
        "field": field,
        "min_val": min_val,
        "max_val": max_val,
        "count": len(results),
        "readings": results,
    })


# ═════════════════════════════════════════════
#  LLM AGENT CHAT  (streaming SSE endpoint for dashboard)
# ═════════════════════════════════════════════

def _init_agent_bridge():
    """Wire the investigation agent's tools to read directly from the
    in-memory data stores, then return the streaming endpoint handler.

    Called once during startup (in main) so the agent doesn't HTTP-call
    itself — it reads nodes/lock/etc. directly via closures.
    """
    try:
        from agent import set_data_accessors, stream_agent_response, get_shared_agent
    except ImportError:
        print("  ⚠  agent.py not found — /api/agent/chat endpoint disabled")
        return False

    # ── Data accessors (closures over the module-level stores) ──────

    def _acc_system_status():
        """Mirror of agent_system_status() logic, returns dict."""
        with lock:
            result = {"maintenance": {}, "power": {}}
            degraded = []
            for agent_type in ["maintenance", "power"]:
                agg = aggregate_stats(agent_type)
                metrics = compute_metrics(agg)
                node_count = len(nodes[agent_type])
                alive = dead = 0
                for nid, store in nodes[agent_type].items():
                    liveness = node_liveness(store)
                    if liveness == "dead":
                        dead += 1
                    else:
                        alive += 1
                    nm = compute_metrics(store["stats"])
                    is_degraded = False
                    reasons = []
                    if liveness == "dead":
                        is_degraded = True
                        reasons.append("node is dead (heartbeat timeout)")
                    if nm["total_readings"] >= 20 and nm["accuracy"] < 0.80:
                        is_degraded = True
                        reasons.append(f"low accuracy ({nm['accuracy']:.1%})")
                    if nm["total_readings"] >= 20 and nm["precision"] > 0 and nm["precision"] < 0.50:
                        is_degraded = True
                        reasons.append(f"high false-positive rate (precision {nm['precision']:.1%})")
                    if nm["total_readings"] >= 20 and nm["recall"] < 0.50:
                        is_degraded = True
                        reasons.append(f"missing detections (recall {nm['recall']:.1%})")
                    if is_degraded:
                        reg = _registered_nodes.get(agent_type, {}).get(nid, {})
                        degraded.append({
                            "node_id": nid, "agent_type": agent_type,
                            "node_name": reg.get("node_name", nid),
                            "liveness": liveness, "accuracy": nm["accuracy"],
                            "precision": nm["precision"], "recall": nm["recall"],
                            "total_readings": nm["total_readings"], "reasons": reasons,
                        })
                result[agent_type] = {
                    "node_count": node_count, "alive": alive, "dead": dead,
                    "total_readings": metrics["total_readings"],
                    "alerts_fired": metrics["alerts_fired"],
                    "accuracy": metrics["accuracy"], "precision": metrics["precision"],
                    "recall": metrics["recall"], "f1_score": metrics["f1_score"],
                }
        total_readings = result["maintenance"]["total_readings"] + result["power"]["total_readings"]
        total_alerts = result["maintenance"]["alerts_fired"] + result["power"]["alerts_fired"]
        if total_readings == 0:
            status = "IDLE"
        elif len(degraded) > 0:
            status = "DEGRADED"
        elif total_alerts > 0:
            status = "ALERT"
        else:
            status = "HEALTHY"
        return {
            "system_status": status, "server_time": datetime.now().isoformat(),
            "total_readings": total_readings, "total_alerts": total_alerts,
            "agents": result, "degraded_nodes": degraded,
        }

    def _acc_node_detail(*, agent_type, node_id, n=20):
        n = min(n, 100)
        with lock:
            store = nodes[agent_type].get(node_id)
            if store is None:
                return {"error": f"Node '{node_id}' not found in '{agent_type}'"}
            reg = _registered_nodes.get(agent_type, {}).get(node_id, {})
            metrics = compute_metrics(store["stats"])
            liveness = node_liveness(store)
            recent_readings = list(store["readings"])[-n:]
            recent_alerts = list(store["alerts"])[-10:]
            elapsed = _time.time() - store["last_heartbeat"]
            trend = {}
            all_readings = list(store["readings"])
            window = all_readings[-50:] if len(all_readings) >= 50 else all_readings
            if len(window) >= 10:
                w_correct = sum(1 for r in window if r.get("prediction") == r.get("actual"))
                w_alerts = sum(1 for r in window if r.get("status") in ("FAILURE", "ATTACK"))
                w_acc = w_correct / len(window)
                w_alert_rate = w_alerts / len(window)
                overall_acc = metrics["accuracy"]
                trend = {
                    "window_size": len(window), "window_accuracy": round(w_acc, 4),
                    "overall_accuracy": overall_acc,
                    "accuracy_delta": round(w_acc - overall_acc, 4),
                    "window_alert_rate": round(w_alert_rate, 4),
                    "trending": "improving" if w_acc > overall_acc + 0.02
                                else "degrading" if w_acc < overall_acc - 0.02
                                else "stable",
                }
        return {
            "node_id": node_id, "agent_type": agent_type,
            "node_name": reg.get("node_name", node_id), "liveness": liveness,
            "registered_at": reg.get("registered_at"),
            "heartbeat_age_s": round(elapsed, 1), "metrics": metrics,
            "trend": trend, "recent_readings": recent_readings,
            "recent_alerts": recent_alerts,
        }

    def _acc_failures(*, agent_type=None, node_id=None, kind="all", n=20):
        n = min(n, 100)
        kind_map = {"missed": "false_negative", "false_alarm": "false_positive"}
        kind = kind_map.get(kind, kind)
        results = []
        with lock:
            for at in ["maintenance", "power"]:
                if agent_type and at != agent_type:
                    continue
                for nid, store in nodes[at].items():
                    if node_id and nid != node_id:
                        continue
                    for r in store["readings"]:
                        pred, actual = r.get("prediction", 0), r.get("actual", 0)
                        if pred == 1 and actual == 1:   ctype = "true_positive"
                        elif pred == 1 and actual == 0: ctype = "false_positive"
                        elif pred == 0 and actual == 1: ctype = "false_negative"
                        else: continue
                        if kind != "all" and ctype != kind:
                            continue
                        entry = dict(r)
                        entry["classification"] = ctype
                        entry["agent_type"] = at
                        entry["node_id_source"] = nid
                        results.append(entry)
        results.sort(key=lambda x: x.get("received_at", x.get("timestamp", "")), reverse=True)
        return {"count": len(results[:n]), "kind_filter": kind, "failures": results[:n]}

    def _acc_sensor_summary(*, agent_type, node_id=None, n=100):
        n = min(n, 500)
        sensor_fields = {
            "maintenance": ["tool_wear", "torque", "power_w", "air_temp",
                            "process_temp", "rotational_speed", "probability"],
            "power": ["r1_vh", "r2_vh", "r3_vh", "r4_vh", "relay_status", "probability"],
        }
        fields = sensor_fields.get(agent_type, [])
        with lock:
            all_readings = []
            for nid, store in nodes[agent_type].items():
                if node_id and nid != node_id:
                    continue
                all_readings.extend(list(store["readings"])[-n:])
        all_readings.sort(key=lambda x: x.get("received_at", x.get("timestamp", "")))
        all_readings = all_readings[-n:]

        def summarise(readings, flds):
            summary = {}
            for f in flds:
                vals = [r[f] for r in readings if f in r and r[f] is not None]
                if not vals: continue
                summary[f] = {
                    "count": len(vals), "min": round(min(vals), 4),
                    "max": round(max(vals), 4), "mean": round(statistics.mean(vals), 4),
                    "std": round(statistics.stdev(vals), 4) if len(vals) >= 2 else 0.0,
                }
            return summary

        alert_key = "FAILURE" if agent_type == "maintenance" else "ATTACK"
        normal = [r for r in all_readings if r.get("status") != alert_key]
        alerts = [r for r in all_readings if r.get("status") == alert_key]
        return {
            "agent_type": agent_type, "node_id": node_id,
            "window_size": len(all_readings),
            "sensors_overall": summarise(all_readings, fields),
            "sensors_normal": summarise(normal, fields),
            "sensors_alert": summarise(alerts, fields),
        }

    def _acc_compare_nodes(*, agent_type):
        node_list = []
        with lock:
            for nid, store in nodes[agent_type].items():
                reg = _registered_nodes.get(agent_type, {}).get(nid, {})
                metrics = compute_metrics(store["stats"])
                total = metrics["total_readings"]
                alert_rate = metrics["alerts_fired"] / total if total > 0 else 0
                node_list.append({
                    "node_id": nid, "node_name": reg.get("node_name", nid),
                    "liveness": node_liveness(store), "total_readings": total,
                    "alerts_fired": metrics["alerts_fired"],
                    "alert_rate": round(alert_rate, 4),
                    "accuracy": metrics["accuracy"], "precision": metrics["precision"],
                    "recall": metrics["recall"], "f1_score": metrics["f1_score"],
                    "confusion_matrix": metrics["confusion_matrix"],
                })
        scored = [n for n in node_list if n["total_readings"] >= 10]
        best = max(scored, key=lambda x: x["f1_score"]) if scored else None
        worst = min(scored, key=lambda x: x["f1_score"]) if scored else None
        return {
            "agent_type": agent_type, "node_count": len(node_list),
            "nodes": node_list,
            "best_node": best["node_id"] if best else None,
            "worst_node": worst["node_id"] if worst else None,
        }

    def _acc_correlation(*, window_s=60, n=20):
        n = min(n, 100)
        with lock:
            m_alerts = []
            for nid, store in nodes["maintenance"].items():
                for a in store["alerts"]:
                    entry = dict(a); entry["_node_id"] = nid; m_alerts.append(entry)
            p_alerts = []
            for nid, store in nodes["power"].items():
                for a in store["alerts"]:
                    entry = dict(a); entry["_node_id"] = nid; p_alerts.append(entry)

        def parse_ts(r):
            ts = r.get("received_at") or r.get("timestamp", "")
            try: return datetime.fromisoformat(ts)
            except (ValueError, TypeError): return None

        correlated = []
        for ma in m_alerts:
            ma_ts = parse_ts(ma)
            if ma_ts is None: continue
            for pa in p_alerts:
                pa_ts = parse_ts(pa)
                if pa_ts is None: continue
                gap = abs((ma_ts - pa_ts).total_seconds())
                if gap <= window_s:
                    correlated.append({
                        "maintenance_alert": {
                            "node_id": ma["_node_id"],
                            "timestamp": ma.get("received_at") or ma.get("timestamp"),
                            "probability": ma.get("probability"), "status": ma.get("status"),
                        },
                        "power_alert": {
                            "node_id": pa["_node_id"],
                            "timestamp": pa.get("received_at") or pa.get("timestamp"),
                            "probability": pa.get("probability"), "status": pa.get("status"),
                        },
                        "time_gap_s": round(gap, 1),
                    })
        correlated.sort(key=lambda x: x["time_gap_s"])
        correlated = correlated[:n]
        total_m, total_p = len(m_alerts), len(p_alerts)
        denom = min(total_m, total_p) if min(total_m, total_p) > 0 else 1
        return {
            "window_s": window_s,
            "summary": {
                "maintenance_alerts_total": total_m, "power_alerts_total": total_p,
                "correlated_count": len(correlated),
                "correlation_pct": round(len(correlated) / denom * 100, 1),
            },
            "correlated_events": correlated,
        }

    def _acc_recent_activity(*, n=30, agent_type=None, alerts_only=False):
        n = min(n, 200)
        events = []
        with lock:
            for at in ["maintenance", "power"]:
                if agent_type and at != agent_type:
                    continue
                alert_key = "FAILURE" if at == "maintenance" else "ATTACK"
                for nid, store in nodes[at].items():
                    for r in store["readings"]:
                        is_alert = r.get("status") == alert_key
                        if alerts_only and not is_alert:
                            continue
                        entry = {
                            "event_type": "alert" if is_alert else "reading",
                            "agent_type": at, "node_id": nid,
                            "timestamp": r.get("received_at") or r.get("timestamp"),
                            "status": r.get("status"), "probability": r.get("probability"),
                            "prediction": r.get("prediction"), "actual": r.get("actual"),
                        }
                        if at == "maintenance":
                            for f in ["tool_wear", "torque", "power_w"]:
                                if f in r: entry[f] = r[f]
                        else:
                            for f in ["r1_vh", "r2_vh", "r3_vh", "r4_vh"]:
                                if f in r: entry[f] = r[f]
                        events.append(entry)
        events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return {"count": len(events[:n]), "events": events[:n]}

    def _acc_search_readings(*, agent_type, field, min_val=None, max_val=None, node_id=None, n=20):
        n = min(n, 100)
        results = []
        with lock:
            for nid, store in nodes[agent_type].items():
                if node_id and nid != node_id:
                    continue
                for r in store["readings"]:
                    val = r.get(field)
                    if val is None: continue
                    if min_val is not None and val < min_val: continue
                    if max_val is not None and val > max_val: continue
                    entry = dict(r); entry["node_id_source"] = nid
                    results.append(entry)
        results.sort(key=lambda x: x.get(field, 0), reverse=True)
        return {
            "agent_type": agent_type, "field": field,
            "min_val": min_val, "max_val": max_val,
            "count": len(results[:n]), "readings": results[:n],
        }

    # ── Register accessors with the agent module ────────────────────

    set_data_accessors({
        "system_status":   _acc_system_status,
        "node_detail":     _acc_node_detail,
        "failures":        _acc_failures,
        "sensor_summary":  _acc_sensor_summary,
        "compare_nodes":   _acc_compare_nodes,
        "correlation":     _acc_correlation,
        "recent_activity": _acc_recent_activity,
        "search_readings": _acc_search_readings,
    })

    # ── Register the SSE endpoint ───────────────────────────────────

    @app.route("/api/agent/chat", methods=["POST"])
    def agent_chat():
        """Streaming chat endpoint for the investigation agent.

        Request JSON:
            message    (str, required): the user's question or instruction
            thread_id  (str, optional): conversation thread ID for memory
                                        (default: "default")

        Response:
            Content-Type: text/event-stream
            SSE events:
                event: token        — streamed text token  {"content": "..."}
                event: tool_call    — agent is calling a tool  {"tool": "...", "args": {...}}
                event: tool_result  — tool returned data  {"tool": "...", "preview": "..."}
                event: done         — stream complete  {"full_response": "..."}
                event: error        — error occurred  {"error": "..."}
        """
        data = request.get_json(force=True)
        if not data or not data.get("message"):
            return jsonify({"error": "message is required"}), 400

        message = data["message"]
        thread_id = data.get("thread_id", "default")

        return Response(
            stream_agent_response(message=message, thread_id=thread_id),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",       # disable nginx buffering
                "Connection": "keep-alive",
            },
        )

    return True


# ═════════════════════════════════════════════
#  OVERVIEW + HEALTH
# ═════════════════════════════════════════════

@app.route("/api/overview", methods=["GET"])
def get_overview():
    """Combined health summary across both agents and all nodes."""
    with lock:
        m_agg = aggregate_stats("maintenance")
        p_agg = aggregate_stats("power")
        m_metrics = compute_metrics(m_agg)
        p_metrics = compute_metrics(p_agg)
        m_node_count = len(nodes["maintenance"])
        p_node_count = len(nodes["power"])

        # Count alive/dead
        m_alive = sum(1 for s in nodes["maintenance"].values() if node_liveness(s) != "dead")
        p_alive = sum(1 for s in nodes["power"].values() if node_liveness(s) != "dead")
        m_dead  = m_node_count - m_alive
        p_dead  = p_node_count - p_alive

    total_readings = m_metrics["total_readings"] + p_metrics["total_readings"]
    total_alerts   = m_metrics["alerts_fired"] + p_metrics["alerts_fired"]

    if total_readings == 0:
        system_status = "IDLE"
    elif total_alerts > 0:
        system_status = "ALERT"
    else:
        system_status = "HEALTHY"

    return jsonify({
        "system_status": system_status,
        "total_readings": total_readings,
        "total_alerts": total_alerts,
        "total_nodes": m_node_count + p_node_count,
        "alive_nodes": m_alive + p_alive,
        "dead_nodes": m_dead + p_dead,
        "agents": {
            "maintenance": {
                "status": "active" if m_metrics["last_updated"] else "idle",
                "node_count": m_node_count,
                "alive": m_alive,
                "dead": m_dead,
                **m_metrics,
            },
            "power": {
                "status": "active" if p_metrics["last_updated"] else "idle",
                "node_count": p_node_count,
                "alive": p_alive,
                "dead": p_dead,
                **p_metrics,
            },
        },
        "heartbeat_timeout_s": HEARTBEAT_TIMEOUT_S,
        "server_time": datetime.now().isoformat(),
    })


@app.route("/api/health", methods=["GET"])
def health():
    with lock:
        m_total = sum(s["stats"]["total"] for s in nodes["maintenance"].values())
        p_total = sum(s["stats"]["total"] for s in nodes["power"].values())
    return jsonify({
        "status": "ok",
        "server_time": datetime.now().isoformat(),
        "maintenance_readings": m_total,
        "power_readings": p_total,
        "maintenance_nodes": len(nodes["maintenance"]),
        "power_nodes": len(nodes["power"]),
        "heartbeat_timeout_s": HEARTBEAT_TIMEOUT_S,
    })


# ═════════════════════════════════════════════
#  RESET ENDPOINT  (for testing)
# ═════════════════════════════════════════════

@app.route("/api/reset", methods=["POST"])
def reset_all():
    """Clear all stored data, registrations and counters. Useful for testing / re-runs."""
    with lock:
        nodes["maintenance"].clear()
        nodes["power"].clear()
        _registered_nodes["maintenance"].clear()
        _registered_nodes["power"].clear()
        _node_counter["maintenance"] = 0
        _node_counter["power"] = 0
    return jsonify({"status": "ok", "message": "All data and registrations cleared"})


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

RESET_C  = "\033[0m"
BOLD_C   = "\033[1m"
CYAN_C   = "\033[36m"
GREEN_C  = "\033[32m"
YELLOW_C = "\033[33m"
DIM_C    = "\033[2m"

def main():
    global HEARTBEAT_TIMEOUT_S

    parser = argparse.ArgumentParser(description="Unified API server for multi-agent anomaly detection dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument("--timeout", type=int, default=30, help="Heartbeat timeout in seconds (default: 30)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument("--no-agent", action="store_true", help="Disable LLM agent chat endpoint")
    args = parser.parse_args()

    HEARTBEAT_TIMEOUT_S = args.timeout

    # ── Initialise the LLM agent bridge ──
    agent_enabled = False
    if not args.no_agent:
        agent_enabled = _init_agent_bridge()

    print(f"\n{BOLD_C}{CYAN_C}{'─'*60}{RESET_C}")
    print(f"{BOLD_C}{CYAN_C}  Multi-Agent Dashboard — API Server{RESET_C}")
    print(f"{BOLD_C}{CYAN_C}{'─'*60}{RESET_C}")
    print(f"{DIM_C}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET_C}\n")
    print(f"  {GREEN_C}Ingest endpoints (must register first):{RESET_C}")
    print(f"    POST  http://{args.host}:{args.port}/api/register")
    print(f"    POST  http://{args.host}:{args.port}/api/maintenance/reading")
    print(f"    POST  http://{args.host}:{args.port}/api/power/reading")
    print(f"\n  {GREEN_C}Heartbeat / liveness:{RESET_C}")
    print(f"    POST  http://{args.host}:{args.port}/api/heartbeat")
    print(f"    DEL   http://{args.host}:{args.port}/api/nodes/<type>/<node_id>")
    print(f"    {DIM_C}Timeout: {args.timeout}s — nodes go DEAD after {args.timeout}s of silence{RESET_C}")
    print(f"\n  {YELLOW_C}Agent tool endpoints (LLM agents via LangChain/LangGraph):{RESET_C}")
    print(f"    GET   http://{args.host}:{args.port}/api/agent/system_status")
    print(f"    GET   http://{args.host}:{args.port}/api/agent/node_detail/<type>/<node_id>")
    print(f"    GET   http://{args.host}:{args.port}/api/agent/failures?kind=false_positive")
    print(f"    GET   http://{args.host}:{args.port}/api/agent/sensor_summary/<type>")
    print(f"    GET   http://{args.host}:{args.port}/api/agent/compare_nodes/<type>")
    print(f"    GET   http://{args.host}:{args.port}/api/agent/correlation")
    print(f"    GET   http://{args.host}:{args.port}/api/agent/recent_activity")
    print(f"    GET   http://{args.host}:{args.port}/api/agent/search_readings?agent_type=maintenance&field=tool_wear&min_val=200")
    if agent_enabled:
        print(f"\n  {YELLOW_C}LLM Agent Chat (streaming SSE):{RESET_C}")
        print(f"    POST  http://{args.host}:{args.port}/api/agent/chat")
        print(f"    {DIM_C}Body: {{\"message\": \"...\", \"thread_id\": \"...\"}}{RESET_C}")
        print(f"    {DIM_C}Returns: text/event-stream (token, tool_call, tool_result, done){RESET_C}")
    else:
        print(f"\n  {DIM_C}LLM Agent Chat: disabled (--no-agent or agent.py not found){RESET_C}")
    print(f"\n  {GREEN_C}Dashboard endpoints:{RESET_C}")
    print(f"    GET   http://{args.host}:{args.port}/api/overview")
    print(f"    GET   http://{args.host}:{args.port}/api/nodes")
    print(f"    GET   http://{args.host}:{args.port}/api/maintenance/latest?n=50&node_id=node_1")
    print(f"    GET   http://{args.host}:{args.port}/api/power/latest?n=50&node_id=node_1")
    print(f"    GET   http://{args.host}:{args.port}/api/maintenance/stats?node_id=node_1")
    print(f"    GET   http://{args.host}:{args.port}/api/power/stats?node_id=node_1")
    print(f"    GET   http://{args.host}:{args.port}/api/maintenance/alerts")
    print(f"    GET   http://{args.host}:{args.port}/api/power/alerts")
    print(f"    GET   http://{args.host}:{args.port}/api/health")
    print(f"    POST  http://{args.host}:{args.port}/api/reset")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()