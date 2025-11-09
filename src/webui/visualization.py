# src/webui/visualization.py

# NOTE: Real-time visualisation server adapted from ShinkaEvolve.
# NOTE: Removes the hard-coded database name dependency.

import http.server
import json
import os
import socketserver
import sqlite3
import time
import urllib.parse
from pathlib import Path

class DatabaseRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler serving the evolution dashboard."""

    # Capture the database path so the handler can stream updates from it.
    def __init__(self, *args, db_path_to_serve=None, **kwargs):
        self.db_path_to_serve = db_path_to_serve
        # Directory containing the static assets (HTML, CSS, JS).
        webui_dir = Path(__file__).parent.resolve()
        super().__init__(*args, directory=str(webui_dir), **kwargs)

    def log_message(self, format, *args):
        # Silence noisy HTTP logs for a cleaner console.
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        
        # Manejar explícitamente la solicitud del favicon para evitar errores
        if parsed_url.path == '/favicon.ico':
            self.send_response(204) # 204 No Content
            self.end_headers()
            return

        if parsed_url.path == "/get_programs":
            return self.handle_get_programs()
        
        if parsed_url.path == "/":
            self.path = "/viz_tree.html"

        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def handle_get_programs(self):
        db_path = self.db_path_to_serve
        if not db_path or not os.path.exists(db_path):
            self.send_error(404, f"Database not found at {db_path}")
            return

        for attempt in range(5):  # Retry up to five times
            try:
                # Open the configured database path directly in read-only mode.
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM concepts")
                rows = cursor.fetchall()
                
                concepts = []
                for row in rows:
                    concept_dict = dict(row)
                    # Deserialize JSON-encoded fields.
                    for key in ['draft_history', 'verification_reports', 'inspiration_ids', 'embedding', 'scores', 'system_requirements']:
                        if key in concept_dict and isinstance(concept_dict[key], str):
                            try:
                                concept_dict[key] = json.loads(concept_dict[key])
                            except json.JSONDecodeError:
                                concept_dict[key] = {} if 'scores' in key or 'req' in key else []
                    concepts.append(concept_dict)
                
                conn.close()
                self.send_json_response(concepts)
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    print(f"  [WebUI] Database is locked, attempt {attempt+1}/5...")
                    time.sleep(0.5 + attempt * 0.5) # Backoff
                else:
                    self.send_error(500, f"Database error: {e}")
                    return
            except Exception as e:
                self.send_error(500, f"An unexpected error occurred: {e}")
                return
        
        self.send_error(503, "Database is busy, please try again later.")

    def send_json_response(self, data):
        payload = json.dumps(data, default=str).encode('utf-8')
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

# Factory that injects the database path into each handler instance.
def create_handler_factory(db_path_to_serve):
    def handler_factory(*args, **kwargs):
        return DatabaseRequestHandler(*args, db_path_to_serve=db_path_to_serve, **kwargs)
    return handler_factory

# Start the visualisation server with the given database path.
def start_server(port: int, db_path_to_serve: str):
    handler_factory = create_handler_factory(db_path_to_serve)
    
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", port), handler_factory) as httpd:
        print(f"[*] Visualisation server running at http://0.0.0.0:{port}")
        print(f"    Serving data from: {db_path_to_serve}")
        httpd.serve_forever()