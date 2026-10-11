"""Keyless official evaluator. Search API exposes validity only, never score."""
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, '/official')
from mlebench.grade import grade_csv, validate_submission
from mlebench.registry import registry
from mlebench.data import is_dataset_prepared

registry = registry.set_data_dir(Path('/dataset'))
if sys.argv[1] == 'grade':
    report = grade_csv(Path('/submission/submission.csv'), registry.get_competition(sys.argv[2]))
    Path('/report/official_report.json').write_text(json.dumps(report.to_dict(), indent=2, default=str))
else:
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.get('/health')
    def health():
        return jsonify(status='running')

    @app.get('/ready')
    def ready():
        task=request.args.get('task')
        if task not in os.environ['ALLOWED_TASKS'].split(','):
            return jsonify(ready=False),400
        try:
            comp=registry.get_competition(task)
            return jsonify(ready=is_dataset_prepared(comp),task=task)
        except Exception as exc:
            return jsonify(ready=False,error_type=type(exc).__name__,task=task)

    @app.post('/validate')
    def validate():
        task = request.headers.get('exp-id')
        if task not in os.environ['ALLOWED_TASKS'].split(',') or 'file' not in request.files:
            return jsonify(error='Invalid task or file'), 400
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'submission.csv'
            request.files['file'].save(path)
            try:
                valid, _message = validate_submission(path, registry.get_competition(task))
                # Do not expose grader exceptions/values derived from private labels.
                return jsonify(is_valid=valid, result='Submission is valid.' if valid else 'Submission format is invalid.')
            except Exception:
                return jsonify(is_valid=False, result='Submission validation failed.')
    app.run(host='127.0.0.1', port=18305)
