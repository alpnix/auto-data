#!/usr/bin/env python3
"""
Minimalist web dashboard for education data pipeline
Shows system status, errors, and query interface
"""

from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from dotenv import load_dotenv
from education_pipeline import EducationDataPipeline
import logging
from collections import deque

load_dotenv()

app = Flask(__name__)
error_log = deque(maxlen=50)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogHandler(logging.Handler):
    """Custom log handler to capture errors"""
    def emit(self, record):
        if record.levelno >= logging.WARNING:
            error_log.append({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'level': record.levelname,
                'message': record.getMessage()
            })


logger.addHandler(LogHandler())


def check_system_status():
    """Check all system components"""
    status = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'components': {}
    }

    brave_key = os.getenv('BRAVE_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')

    status['components']['brave_api'] = {
        'name': 'Brave Search API',
        'status': 'active' if brave_key else 'inactive',
        'message': 'Configured' if brave_key else 'Missing BRAVE_KEY'
    }

    status['components']['openai_api'] = {
        'name': 'OpenAI API (gpt-4o-mini)',
        'status': 'active' if openai_key else 'inactive',
        'message': 'Configured' if openai_key else 'Missing OPENAI_API_KEY'
    }

    try:
        from sentence_transformers import SentenceTransformer
        status['components']['embeddings'] = {
            'name': 'Embeddings (BGE-small)',
            'status': 'active',
            'message': 'Local model ready'
        }
    except Exception as e:
        status['components']['embeddings'] = {
            'name': 'Embeddings',
            'status': 'error',
            'message': str(e)
        }

    try:
        from pymilvus import connections
        connections.connect("health_check", uri="./milvus_education.db")
        connections.disconnect("health_check")
        status['components']['milvus'] = {
            'name': 'Milvus Vector DB',
            'status': 'active',
            'message': 'Milvus Lite connected'
        }
    except Exception as e:
        status['components']['milvus'] = {
            'name': 'Milvus Vector DB',
            'status': 'error',
            'message': str(e)[:100]
        }

    return status


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """Return system status as JSON"""
    return jsonify(check_system_status())


@app.route('/api/errors')
def api_errors():
    """Return recent errors"""
    return jsonify(list(error_log))


@app.route('/api/query', methods=['POST'])
def api_query():
    """Process a query"""
    try:
        data = request.json
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        pipeline = EducationDataPipeline()
        pipeline.initialize()

        try:
            result = pipeline.process_query(query=query)

            return jsonify({
                'success': True,
                'query': result['query'],
                'results_count': result['search_results_count'],
                'answer': result['llm_decision'],
                'sources': [
                    {
                        'url': doc['url'],
                        'score': round(doc['score'], 3),
                        'text': doc['text'][:200] + '...'
                    }
                    for doc in result['relevant_docs'][:3]
                ]
            })

        finally:
            pipeline.cleanup()

    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Education Pipeline Dashboard")
    print("="*60)
    print("\nStarting server at http://localhost:8080")
    print("Press Ctrl+C to stop\n")

    app.run(debug=True, host='0.0.0.0', port=8080)
