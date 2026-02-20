#!/usr/bin/env python3
"""
Setup Server - Serves both the agentic setup UI and research dashboard
Combined server for the complete workflow:
1. Setup: Drop files → Agent configures pipeline
2. Research: Run pipeline → View results with evaluation
"""

from flask import Flask, render_template, Response, jsonify, request
import json
import time
from research_pipeline import ResearchPipeline, get_available_models, DEFAULT_MODEL
from results_manager import ResultsManager
from agentic_setup import AgenticSetup
from evaluation import Evaluator, QA_METRICS_AVAILABLE
from config_helpers import get_entity_primary_value, get_entity_display

app = Flask(__name__)

# Initialize components
agent = AgenticSetup()
evaluator = Evaluator() if QA_METRICS_AVAILABLE else None


# =============================================================================
# SETUP ROUTES (Agentic Configuration)
# =============================================================================

@app.route('/')
def index():
    """Landing page — project overview"""
    return render_template('overview.html')


@app.route('/overview')
def overview_page():
    """Project overview / presentation page"""
    return render_template('overview.html')


@app.route('/setup')
def setup_page():
    """Setup page for dropping files"""
    return render_template('setup.html')


@app.route('/api/setup/count', methods=['POST'])
def count_rows():
    """Count rows in uploaded CSV"""
    try:
        data = request.json
        dataset = data.get('dataset', '')

        if not dataset:
            return jsonify({'count': 0})

        count = agent.count_csv_rows(dataset)
        return jsonify({'count': count})

    except Exception as e:
        return jsonify({'count': 0, 'error': str(e)})


@app.route('/api/setup/generate', methods=['POST'])
def generate_config():
    """Generate configuration from uploaded files"""
    try:
        data = request.json
        instructions = data.get('instructions', '')
        dataset = data.get('dataset', '')
        extra = data.get('extra', '')
        start_row = data.get('start_row', 0)
        end_row = data.get('end_row', None)
        selected_model = data.get('model', DEFAULT_MODEL)

        if not instructions or not dataset:
            return jsonify({
                'success': False,
                'error': 'Both instructions and dataset are required',
                'logs': ['[ERROR] Missing required input files']
            })

        # Create a fresh agent instance per request so model choice is respected
        request_agent = AgenticSetup(model=selected_model)

        result = request_agent.process(
            dataset_csv=dataset,
            instructions_text=instructions,
            extra_instructions=extra,
            auto_write=False,
            start_row=start_row,
            end_row=end_row
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'logs': [f'[ERROR] {str(e)}']
        })


@app.route('/api/setup/apply', methods=['POST'])
def apply_config():
    """Apply generated configuration"""
    try:
        data = request.json
        config = data.get('config', '')

        if not config:
            return jsonify({
                'success': False,
                'error': 'No configuration provided'
            })

        success, message = agent.write_config(config)

        return jsonify({
            'success': success,
            'message': message
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


# =============================================================================
# RESEARCH ROUTES (Pipeline Execution)
# =============================================================================

@app.route('/research')
def research_page():
    """Research dashboard page"""
    return render_template('research.html')


@app.route('/api/models')
def list_models():
    """Return available LLM models (those with API keys configured)"""
    models = get_available_models()
    return jsonify({'models': models, 'default': DEFAULT_MODEL})


@app.route('/api/config')
def get_config():
    """Return current pipeline configuration"""
    try:
        # Dynamically import to get latest config
        import importlib
        import pipeline_config
        importlib.reload(pipeline_config)
        
        # Get entity schema with fallback
        entity_schema = getattr(pipeline_config, 'ENTITY_SCHEMA', None)
        if not entity_schema:
            # Infer from TARGET_ENTITIES
            entities = getattr(pipeline_config, 'TARGET_ENTITIES', [])
            if entities and isinstance(entities[0], dict):
                keys = list(entities[0].keys())
                # Prefer 'name' as primary field
                primary = next((k for k in keys if 'name' in k.lower()), keys[0])
                entity_schema = {"primary_field": primary, "fields": keys}
            else:
                entity_schema = {"primary_field": "name", "fields": ["name"]}

        return jsonify({
            'dataset_name': getattr(pipeline_config, 'DATASET_NAME', 'Unknown'),
            'dataset_description': getattr(pipeline_config, 'DATASET_DESCRIPTION', ''),
            'fields': getattr(pipeline_config, 'EXTRACTION_FIELDS', []),
            'entities': getattr(pipeline_config, 'TARGET_ENTITIES', []),
            'entity_schema': entity_schema,
            'ground_truth': getattr(pipeline_config, 'GROUND_TRUTH', None),
            'has_evaluator': evaluator is not None
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'fields': [],
            'entities': [],
            'entity_schema': {"primary_field": "name", "fields": ["name"]}
        })


@app.route('/api/research/process')
def research_process():
    """Server-Sent Events endpoint — delegates all logic to process_entity()."""
    selected_model = request.args.get('model', DEFAULT_MODEL)

    def generate():
        try:
            import importlib
            import pipeline_config
            importlib.reload(pipeline_config)

            entities = getattr(pipeline_config, 'TARGET_ENTITIES', [])
            fields   = getattr(pipeline_config, 'EXTRACTION_FIELDS', [])
            ground_truth = getattr(pipeline_config, 'GROUND_TRUTH', None)

            if not entities:
                yield f"event: error\ndata: {json.dumps({'error': 'No entities configured'})}\n\n"
                return

            yield f"event: config\ndata: {json.dumps({'entities': entities, 'fields': fields, 'model': selected_model})}\n\n"

            pipeline = ResearchPipeline(model=selected_model)
            results_manager = ResultsManager()

            try:
                session_folder = results_manager.create_session_folder()
                start_time = time.time()
                pipeline.initialize()

                all_results = []
                total_documents = 0
                unique_sources = set()
                log_entries = []

                for idx, entity in enumerate(entities):
                    yield f"event: start\ndata: {json.dumps({'index': idx, 'entity': entity})}\n\n"

                    # on_event callback forwards pipeline events to SSE stream
                    def on_event(event_type, data, _idx=idx):
                        data['index'] = _idx
                        return event_type, data

                    sse_queue = []

                    def collect_event(event_type, data, _idx=idx):
                        data['index'] = _idx
                        sse_queue.append((event_type, data))

                    # Run full pipeline for this entity
                    result = pipeline.process_entity(entity, on_event=collect_event)

                    # Flush queued SSE events
                    for event_type, data in sse_queue:
                        yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

                    extracted_info = result['extracted_info']
                    entity_id = get_entity_primary_value(entity, pipeline_config)

                    # Evaluate against ground truth if available
                    eval_result = None
                    if evaluator and ground_truth and entity_id in ground_truth:
                        eval_result = evaluator.evaluate_entity(
                            entity_name=entity_id,
                            extracted_data=extracted_info,
                            ground_truth=ground_truth[entity_id],
                            fields_config=fields
                        )

                    all_results.append(result)
                    total_documents += result['documents_collected']
                    for src in result.get('sources', []):
                        unique_sources.add(src.get('url', ''))

                    # Send completion event
                    try:
                        complete_data = {
                            'index': idx,
                            'documents': result['documents_collected'],
                            'extracted_info': extracted_info,
                            'evaluation': eval_result
                        }
                        yield f"event: complete\ndata: {json.dumps(complete_data)}\n\n"
                    except (TypeError, ValueError):
                        yield f"event: complete\ndata: {json.dumps({'index': idx, 'documents': result['documents_collected'], 'extracted_info': {'error': 'Serialization failed'}, 'evaluation': None})}\n\n"

                    log_entries.append(f"PROCESSED: {get_entity_display(entity, pipeline_config)}")

                    completion = int(((idx + 1) / len(entities)) * 100)
                    yield f"event: stats\ndata: {json.dumps({'total_documents': total_documents, 'unique_sources': len(unique_sources), 'completion': completion})}\n\n"

                # Save session
                end_time = time.time()
                stats = {
                    'total_entities': len(all_results),
                    'total_documents': total_documents,
                    'unique_sources': len(unique_sources),
                    'total_searches': pipeline.search_count,
                    'processing_time': f"{end_time - start_time:.1f} seconds"
                }
                saved_files = results_manager.save_results(all_results, session_folder)
                results_manager.save_stats(stats, session_folder)
                results_manager.save_session_log(log_entries, session_folder)
                results_manager.create_summary_report(all_results, stats, session_folder)

                yield f"event: done\ndata: {json.dumps({'total': len(entities), 'searches': pipeline.search_count, 'session': str(session_folder)})}\n\n"
                yield f"event: saved\ndata: {json.dumps({'session_folder': str(session_folder), 'files': saved_files})}\n\n"

            finally:
                pipeline.cleanup()

        except Exception as e:
            import traceback
            print(f"[ERROR] SSE Stream error: {e}\n{traceback.format_exc()}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/sessions')
def list_sessions():
    """List all saved research sessions"""
    results_manager = ResultsManager()
    sessions = results_manager.list_sessions()
    return jsonify(sessions)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Research Pipeline Server")
    print("="*60)
    print(f"\nEvaluator: {'F1/EM (qa-metrics)' if evaluator else 'Basic'}")
    print("\nRoutes:")
    print("  /         - Setup (drop files)")
    print("  /setup    - Setup (drop files)")
    print("  /research - Research dashboard")
    print("\nStarting server at http://localhost:8080")
    print("Press Ctrl+C to stop\n")

    app.run(debug=False, host='0.0.0.0', port=8080, threaded=True)
