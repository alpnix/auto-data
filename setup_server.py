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
from faculty_research_pipeline import ResearchPipeline
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
    """Landing page - redirect to setup"""
    return render_template('setup.html')


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

        if not instructions or not dataset:
            return jsonify({
                'success': False,
                'error': 'Both instructions and dataset are required',
                'logs': ['[ERROR] Missing required input files']
            })

        result = agent.process(
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


@app.route('/api/config')
def get_config():
    """Return current pipeline configuration"""
    try:
        # Dynamically import to get latest config
        import importlib
        import pipeline_config
        importlib.reload(pipeline_config)

        return jsonify({
            'dataset_name': getattr(pipeline_config, 'DATASET_NAME', 'Unknown'),
            'dataset_description': getattr(pipeline_config, 'DATASET_DESCRIPTION', ''),
            'fields': getattr(pipeline_config, 'EXTRACTION_FIELDS', []),
            'entities': getattr(pipeline_config, 'TARGET_ENTITIES', []),
            'ground_truth': getattr(pipeline_config, 'GROUND_TRUTH', None),
            'has_evaluator': evaluator is not None
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'fields': [],
            'entities': []
        })


@app.route('/api/research/process')
def research_process():
    """Server-Sent Events endpoint for research process"""

    def generate():
        """Generate SSE events for the research process"""
        try:
            # Dynamically import to get latest config
            import importlib
            import pipeline_config
            importlib.reload(pipeline_config)

            entities = getattr(pipeline_config, 'TARGET_ENTITIES', [])
            fields = getattr(pipeline_config, 'EXTRACTION_FIELDS', [])
            ground_truth = getattr(pipeline_config, 'GROUND_TRUTH', None)

            if not entities:
                yield f"event: error\ndata: {json.dumps({'error': 'No entities configured'})}\n\n"
                return

            # Send config info
            yield f"event: config\ndata: {json.dumps({'entities': entities, 'fields': fields})}\n\n"

            pipeline = ResearchPipeline()
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
                    # Send start event
                    yield f"event: start\ndata: {json.dumps({'index': idx, 'entity': entity})}\n\n"

                    # Generate search queries
                    search_queries = pipeline.generate_search_queries(entity)
                    total_queries = len(search_queries)

                    all_documents = []
                    texts_to_store = []
                    urls_to_store = []
                    query_types_to_store = []

                    # Execute searches
                    for q_idx, query_info in enumerate(search_queries):
                        query = query_info['query']
                        query_type = query_info['type']

                        progress = int((q_idx / total_queries) * 70)
                        yield f"event: search\ndata: {json.dumps({'index': idx, 'description': query_info['description'], 'progress': progress, 'total_searches': pipeline.search_count})}\n\n"

                        search_results = pipeline.search_google(query)
                        time.sleep(1)

                        if search_results:
                            yield f"event: result\ndata: {json.dumps({'index': idx, 'count': len(search_results), 'query_type': query_type})}\n\n"

                        for result in search_results:
                            text = f"{result['title']}. {result['description']}"
                            url = result['url']

                            texts_to_store.append(text)
                            urls_to_store.append(url)
                            query_types_to_store.append(query_type)

                            all_documents.append({
                                "query_type": query_type,
                                "url": url,
                                "text": text
                            })

                            unique_sources.add(url)

                            # Scrape key URLs
                            if query_type in ['profile', 'cv', 'site_specific'] and len([d for d in all_documents if d['query_type'].endswith('_full')]) < 3:
                                yield f"event: scraping\ndata: {json.dumps({'index': idx, 'url': url[:50]})}\n\n"
                                scraped = pipeline.scrape_content(url)
                                if scraped:
                                    texts_to_store.append(scraped)
                                    urls_to_store.append(url)
                                    query_types_to_store.append(f"{query_type}_full")
                                    all_documents.append({
                                        "query_type": f"{query_type}_full",
                                        "url": url,
                                        "text": scraped
                                    })

                    # Store in vector database
                    if texts_to_store:
                        yield f"event: storing\ndata: {json.dumps({'index': idx, 'count': len(texts_to_store)})}\n\n"

                        entity_id = get_entity_primary_value(entity, pipeline_config)
                        entity_meta = json.dumps(entity)

                        pipeline.store_in_vector_db(
                            entity_id,
                            entity_meta,
                            texts_to_store,
                            urls_to_store,
                            query_types_to_store
                        )

                    # Extract information
                    yield f"event: extracting\ndata: {json.dumps({'index': idx})}\n\n"

                    extracted_info = pipeline.extract_entity_info(
                        entity,
                        all_documents[:8]
                    )

                    # Evaluate if ground truth available
                    eval_result = None
                    entity_id = get_entity_primary_value(entity, pipeline_config)
                    if evaluator and ground_truth and entity_id in ground_truth:
                        eval_result = evaluator.evaluate_entity(
                            entity_name=entity_id,
                            extracted_data=extracted_info,
                            ground_truth=ground_truth[entity_id],
                            fields_config=fields
                        )

                    # Store result
                    result = {
                        "entity": entity,
                        "documents_collected": len(all_documents),
                        "sources": all_documents,
                        "extracted_info": extracted_info
                    }
                    all_results.append(result)
                    total_documents += len(all_documents)

                    # Send completion (with safe JSON serialization)
                    try:
                        complete_data = {
                            'index': idx,
                            'documents': len(all_documents),
                            'extracted_info': extracted_info,
                            'evaluation': eval_result
                        }
                        yield f"event: complete\ndata: {json.dumps(complete_data)}\n\n"
                    except (TypeError, ValueError) as json_err:
                        # Fallback if JSON serialization fails
                        print(f"[WARN] JSON serialization error: {json_err}")
                        safe_data = {
                            'index': idx,
                            'documents': len(all_documents),
                            'extracted_info': {'error': 'Serialization failed'},
                            'evaluation': None
                        }
                        yield f"event: complete\ndata: {json.dumps(safe_data)}\n\n"

                    # Log
                    log_entries.append(f"PROCESSED: {get_entity_display(entity, pipeline_config)}")

                    # Stats update
                    completion = int(((idx + 1) / len(entities)) * 100)
                    yield f"event: stats\ndata: {json.dumps({'total_documents': total_documents, 'unique_sources': len(unique_sources), 'completion': completion})}\n\n"

                # Save results
                end_time = time.time()
                processing_time = end_time - start_time

                stats = {
                    'total_entities': len(all_results),
                    'total_documents': total_documents,
                    'unique_sources': len(unique_sources),
                    'total_searches': pipeline.search_count,
                    'processing_time': f"{processing_time:.1f} seconds"
                }

                saved_files = results_manager.save_results(all_results, session_folder)
                results_manager.save_stats(stats, session_folder)
                results_manager.save_session_log(log_entries, session_folder)
                results_manager.create_summary_report(all_results, stats, session_folder)

                # Send done
                yield f"event: done\ndata: {json.dumps({'total': len(entities), 'searches': pipeline.search_count, 'session': str(session_folder)})}\n\n"

                yield f"event: saved\ndata: {json.dumps({'session_folder': str(session_folder), 'files': saved_files})}\n\n"

            finally:
                pipeline.cleanup()

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[ERROR] SSE Stream error: {e}")
            print(error_trace)
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
