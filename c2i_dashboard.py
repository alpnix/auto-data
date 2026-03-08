#!/usr/bin/env python3
"""
Research Dashboard with Real-Time Process Visualization
Shows live progress of data collection using Server-Sent Events.
Dataset-agnostic: reads entity/field definitions from pipeline_config.
"""

from flask import Flask, render_template, Response, jsonify
import json
import time
from research_pipeline import ResearchPipeline
from results_manager import ResultsManager
from evaluation import Evaluator, QA_METRICS_AVAILABLE
from config_helpers import load_config, get_entity_primary_value, get_entity_display
import importlib

app = Flask(__name__)

# Initialize evaluator
evaluator = Evaluator() if QA_METRICS_AVAILABLE else None


@app.route('/')
def index():
    """Process visualization page"""
    return render_template('research_process.html')


@app.route('/api/research/process')
def research_process():
    """Server-Sent Events endpoint for real-time process updates"""

    def generate():
        """Generate SSE events for the research process"""

        # Load config dynamically
        config = load_config()
        if not config:
            yield f"event: error\ndata: {json.dumps({'error': 'No pipeline_config found'})}\n\n"
            return

        entities = getattr(config, 'TARGET_ENTITIES', [])
        fields = getattr(config, 'EXTRACTION_FIELDS', [])
        ground_truth = getattr(config, 'GROUND_TRUTH', None)

        if not entities:
            yield f"event: error\ndata: {json.dumps({'error': 'No entities configured'})}\n\n"
            return

        # Send config info so the UI can render dynamically
        yield f"event: config\ndata: {json.dumps({'entities': entities, 'fields': fields, 'ground_truth': ground_truth})}\n\n"

        pipeline = ResearchPipeline()
        results_manager = ResultsManager()

        try:
            # Create session folder for this run
            session_folder = results_manager.create_session_folder()
            start_time = time.time()

            pipeline.initialize()

            all_results = []
            total_documents = 0
            unique_sources = set()
            log_entries = []

            for idx, entity in enumerate(entities):
                entity_id = get_entity_primary_value(entity, config)

                # Send start event
                yield f"event: start\ndata: {json.dumps({'index': idx, 'name': entity_id, 'entity': entity})}\n\n"

                # Generate search queries
                search_queries = pipeline.generate_search_queries(entity)
                total_queries = len(search_queries)

                all_documents = []
                texts_to_store = []
                urls_to_store = []
                query_types_to_store = []

                # Execute searches with progress updates
                for q_idx, query_info in enumerate(search_queries):
                    query = query_info['query']
                    query_type = query_info['type']

                    # Send search event
                    progress = int((q_idx / total_queries) * 70)  # 70% for searching
                    yield f"event: search\ndata: {json.dumps({'index': idx, 'description': query_info['description'], 'progress': progress, 'total_searches': pipeline.search_count})}\n\n"

                    search_results = pipeline.search_google(query)
                    time.sleep(1)  # Rate limiting

                    if search_results:
                        # Send result event
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

                        # Try to scrape full content for key URLs
                        if query_type in ['profile', 'cv', 'site_specific'] and len([d for d in all_documents if d['query_type'].endswith('_full')]) < 3:
                            yield f"event: scraping\ndata: {json.dumps({'index': idx, 'url': url})}\n\n"
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
                    progress = 75
                    yield f"event: search\ndata: {json.dumps({'index': idx, 'description': 'Storing documents', 'progress': progress, 'total_searches': pipeline.search_count})}\n\n"
                    yield f"event: storing\ndata: {json.dumps({'index': idx, 'count': len(texts_to_store)})}\n\n"

                    entity_meta = json.dumps(entity)
                    pipeline.store_in_vector_db(
                        entity_id,
                        entity_meta,
                        texts_to_store,
                        urls_to_store,
                        query_types_to_store
                    )

                # Extract structured information
                progress = 85
                yield f"event: search\ndata: {json.dumps({'index': idx, 'description': 'Extracting information', 'progress': progress, 'total_searches': pipeline.search_count})}\n\n"
                yield f"event: extracting\ndata: {json.dumps({'index': idx})}\n\n"

                extracted_info = pipeline.extract_entity_info(
                    entity,
                    all_documents[:8]
                )

                # Store results
                result = {
                    "entity": entity,
                    "documents_collected": len(all_documents),
                    "sources": all_documents,
                    "extracted_info": extracted_info
                }
                all_results.append(result)
                total_documents += len(all_documents)

                # Evaluate against ground truth if available
                eval_result = None
                if evaluator and ground_truth and entity_id in ground_truth:
                    eval_result = evaluator.evaluate_entity(
                        entity_name=entity_id,
                        extracted_data=extracted_info,
                        ground_truth=ground_truth[entity_id],
                        fields_config=fields
                    )

                # Send completion event with evaluation
                yield f"event: complete\ndata: {json.dumps({'index': idx, 'documents': len(all_documents), 'extracted_info': extracted_info, 'evaluation': eval_result})}\n\n"

                # Add to log
                log_entries.append(f"\n{'='*80}")
                log_entries.append(f"PROCESSED: {get_entity_display(entity, config)}")
                log_entries.append(f"{'='*80}")
                log_entries.append(f"Documents collected: {len(all_documents)}")

                # Send stats update
                completion = int(((idx + 1) / len(entities)) * 100)
                yield f"event: stats\ndata: {json.dumps({'total_documents': total_documents, 'unique_sources': len(unique_sources), 'completion': completion})}\n\n"

            # Calculate final stats
            end_time = time.time()
            processing_time = end_time - start_time

            stats = {
                'total_entities': len(all_results),
                'total_documents': total_documents,
                'unique_sources': len(unique_sources),
                'total_searches': pipeline.search_count,
                'processing_time': f"{processing_time:.1f} seconds",
                'processing_time_seconds': processing_time,
                'avg_time_per_entity': f"{processing_time / len(all_results):.1f} seconds"
            }

            # Save all results
            saved_files = results_manager.save_results(all_results, session_folder)
            stats_file = results_manager.save_stats(stats, session_folder)
            log_file = results_manager.save_session_log(log_entries, session_folder)
            summary_file = results_manager.create_summary_report(all_results, stats, session_folder)

            # Send final done event with session info
            yield f"event: done\ndata: {json.dumps({'total_entities': len(entities), 'total_searches': pipeline.search_count, 'session_folder': str(session_folder)})}\n\n"

            # Send saved files info
            yield f"event: saved\ndata: {json.dumps({'session_folder': str(session_folder), 'files': saved_files, 'stats_file': stats_file, 'summary_file': summary_file})}\n\n"

        finally:
            pipeline.cleanup()

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/sessions')
def list_sessions():
    """List all saved research sessions"""
    results_manager = ResultsManager()
    sessions = results_manager.list_sessions()
    return jsonify(sessions)


@app.route('/api/config')
def get_config():
    """Return pipeline configuration for dynamic UI"""
    config = load_config()
    if not config:
        return jsonify({'error': 'No pipeline_config found', 'fields': [], 'entities': []})

    return jsonify({
        'fields': getattr(config, 'EXTRACTION_FIELDS', []),
        'ground_truth': getattr(config, 'GROUND_TRUTH', None),
        'entities': getattr(config, 'TARGET_ENTITIES', []),
        'entity_schema': getattr(config, 'ENTITY_SCHEMA', None),
        'has_evaluator': evaluator is not None
    })


if __name__ == '__main__':
    config = load_config()
    entities = getattr(config, 'TARGET_ENTITIES', []) if config else []

    print("\n" + "="*60)
    print("Research Dashboard")
    print("="*60)
    print(f"\nEvaluator: {'F1/EM (qa-metrics)' if evaluator else 'Basic string match'}")
    print(f"\nTarget Entities (from config): {len(entities)}")
    for i, entity in enumerate(entities):
        print(f"  {i}. {get_entity_display(entity, config)}")
    print("\nStarting server at http://localhost:8081")
    print("Press Ctrl+C to stop\n")

    app.run(debug=False, host='0.0.0.0', port=8081, threaded=True)
