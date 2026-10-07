"""
Flask server. Run from the project root:  python app.py  ->  http://localhost:8080

Pages
  /            overview.html   project overview
  /setup       setup.html      drop CSV + instructions, generate and save pipeline_config.py
  /research    research.html   run the pipeline with live progress and evaluation

API
  GET  /api/models            models with an API key configured (+ default)
  POST /api/setup/generate    {instructions, dataset, extra, start_row, end_row, model} -> {success, config, logs}
  POST /api/setup/apply       {config} -> write pipeline_config.py
  GET  /api/config            current pipeline_config contents for the dashboard
  GET  /api/research/process  Server-Sent Events stream of one full run (?model=)
  GET  /api/sessions          saved runs under research_results/
"""

import json
import time

from flask import Flask, Response, jsonify, render_template, request

from core.agentic_setup import AgenticSetup
from core.config_helpers import get_entity_display, get_entity_primary_value, get_entity_schema, load_config
from core.evaluation import Evaluator
from core.llm import DEFAULT_MODEL, get_available_models
from core.research_pipeline import ResearchPipeline
from core.results_manager import ResultsManager

app = Flask(__name__)
evaluator = Evaluator()


def sse(event: str, data) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# ── pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def overview():
    return render_template("overview.html")


@app.route("/setup")
def setup_page():
    return render_template("setup.html")


@app.route("/research")
def research_page():
    return render_template("research.html")


# ── setup API ────────────────────────────────────────────────────────────────

@app.route("/api/models")
def list_models():
    return jsonify({"models": get_available_models(), "default": DEFAULT_MODEL})


@app.route("/api/setup/generate", methods=["POST"])
def generate_config():
    body = request.json or {}
    if not body.get("instructions") or not body.get("dataset"):
        return jsonify({"success": False, "error": "Both instructions and dataset are required", "logs": []})
    agent = AgenticSetup(model=body.get("model") or DEFAULT_MODEL)
    return jsonify(agent.process(
        dataset_csv=body["dataset"],
        instructions_text=body["instructions"],
        extra=body.get("extra", ""),
        start_row=int(body.get("start_row") or 0),
        end_row=body.get("end_row"),
    ))


@app.route("/api/setup/apply", methods=["POST"])
def apply_config():
    code = (request.json or {}).get("config", "")
    if not code:
        return jsonify({"success": False, "error": "No configuration provided"})
    ok, message = AgenticSetup.write_config(code)
    return jsonify({"success": ok, "message": message, "error": None if ok else message})


# ── research API ─────────────────────────────────────────────────────────────

@app.route("/api/config")
def get_config():
    config = load_config()
    if not config:
        return jsonify({"error": "pipeline_config.py not found or invalid", "entities": [], "fields": []})
    return jsonify({
        "dataset_name": getattr(config, "DATASET_NAME", ""),
        "dataset_description": getattr(config, "DATASET_DESCRIPTION", ""),
        "entity_schema": get_entity_schema(config),
        "entities": getattr(config, "TARGET_ENTITIES", []),
        "fields": getattr(config, "EXTRACTION_FIELDS", []),
        "ground_truth": getattr(config, "GROUND_TRUTH", {}),
    })


@app.route("/api/research/process")
def research_process():
    model = request.args.get("model", DEFAULT_MODEL)

    def stream():
        config = load_config()
        entities = getattr(config, "TARGET_ENTITIES", []) if config else []
        fields = getattr(config, "EXTRACTION_FIELDS", []) if config else []
        ground_truth = getattr(config, "GROUND_TRUTH", {}) if config else {}
        if not entities:
            yield sse("error", {"error": "No entities configured — run Setup first"})
            return

        pipeline, manager = ResearchPipeline(model=model), ResultsManager()
        results, sources, started = [], set(), time.time()
        try:
            pipeline.initialize()
            folder = manager.create_session_folder()
            for idx, entity in enumerate(entities):
                yield sse("start", {"index": idx, "entity": entity})

                # process_entity() is synchronous, so progress events are buffered and flushed after.
                events = []
                result = pipeline.process_entity(entity, on_event=lambda t, d: events.append((t, {**d, "index": idx})))
                for event_type, data in events:
                    yield sse(event_type, data)

                entity_id = get_entity_primary_value(entity, config)
                evaluation = None
                if entity_id in ground_truth and "error" not in result["extracted_info"]:
                    evaluation = evaluator.evaluate_entity(entity_id, result["extracted_info"], ground_truth[entity_id], fields)

                results.append(result)
                sources.update(s["url"] for s in result["sources"])
                yield sse("complete", {"index": idx, "documents": result["documents_collected"],
                                       "extracted_info": result["extracted_info"], "evaluation": evaluation})
                yield sse("stats", {"total_documents": sum(r["documents_collected"] for r in results),
                                    "unique_sources": len(sources),
                                    "completion": int((idx + 1) / len(entities) * 100)})

            stats = {"total_entities": len(results),
                     "total_documents": sum(r["documents_collected"] for r in results),
                     "unique_sources": len(sources),
                     "total_searches": pipeline.search_count,
                     "processing_time": f"{time.time() - started:.1f} seconds"}
            files = manager.save_session(results, stats, folder)
            yield sse("done", {"total": len(results), "searches": pipeline.search_count})
            yield sse("saved", {"session_folder": str(folder), "files": files})
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield sse("error", {"error": str(e)})
        finally:
            pipeline.cleanup()

    return Response(stream(), mimetype="text/event-stream")


@app.route("/api/sessions")
def list_sessions():
    return jsonify(ResultsManager().list_sessions())


if __name__ == "__main__":
    print("Research pipeline server → http://localhost:8080  (/setup, /research)")
    app.run(host="0.0.0.0", port=8080, threaded=True)
