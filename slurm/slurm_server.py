import os
import sys
import json
import urllib.parse
import subprocess
import socketserver
import getpass
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8081  # Use 8081 to avoid conflict with the robot server on 8080
active_process = None

# Automatically resolve paths relative to script location
SLURM_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCAL_DIR = os.path.dirname(SLURM_DIR) + "/"
DEFAULT_REMOTE_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/SwiftSketch-Protraitron/"
PASSWORD_FILE = os.path.expanduser("~/.ssh/.tau_password")
CLUSTER_USER = "avnerf"
CLUSTER_HOST = "slurm-client.cs.tau.ac.il"

def get_password():
    if os.path.exists(PASSWORD_FILE):
        try:
            with open(PASSWORD_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"[WARNING] Failed to read password file: {e}")
    return None

class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """
    Multi-threaded HTTP server using standard Python libraries.
    Allows concurrent requests, e.g. aborting sync/sbatch while streaming logs.
    """
    daemon_threads = True

class SlurmHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence default logging to keep terminal clean
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        if path == '/' or path == '/index.html':
            self.serve_html()
        elif path == '/api/squeue':
            self.handle_squeue()
        elif path == '/api/run':
            self.handle_run_stream(parsed_url.query)
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        if path == '/api/abort':
            self.handle_abort()
        elif path == '/api/scancel':
            self.handle_scancel(parsed_url.query)
        else:
            self.send_error(404, "Endpoint not found")

    def serve_html(self):
        try:
            html_path = os.path.join(SLURM_DIR, 'slurm_dashboard.html')
            if not os.path.exists(html_path):
                self.send_error(404, "slurm_dashboard.html not found")
                return
                
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(content.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except Exception as e:
            self.send_error(500, f"Error serving HTML: {e}")

    def run_remote_cmd(self, command):
        """Helper to run a remote command synchronously and return stdout/stderr."""
        password = get_password()
        full_command = f"cd {DEFAULT_REMOTE_DIR} && {command}"
        
        if password:
            args = [
                "sshpass", "-e",
                "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                f"{CLUSTER_USER}@{CLUSTER_HOST}",
                full_command
            ]
            env = os.environ.copy()
            env["SSHPASS"] = password
        else:
            args = [
                "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                f"{CLUSTER_USER}@{CLUSTER_HOST}",
                full_command
            ]
            env = None
            
        result = subprocess.run(args, capture_output=True, text=True, env=env)
        return result.returncode, result.stdout, result.stderr

    def handle_squeue(self):
        code, stdout, stderr = self.run_remote_cmd(f"squeue -u {CLUSTER_USER}")
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        if code == 0:
            self.wfile.write(json.dumps({"status": "success", "output": stdout}).encode('utf-8'))
        else:
            self.wfile.write(json.dumps({"status": "error", "error": stderr}).encode('utf-8'))

    def handle_scancel(self, query_string):
        params = urllib.parse.parse_qs(query_string)
        job_id = params.get('job_id', [''])[0]
        if not job_id:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing job_id parameter")
            return
            
        cmd = f"scancel {job_id}"
        code, stdout, stderr = self.run_remote_cmd(cmd)
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        if code == 0:
            self.wfile.write(json.dumps({"status": "success", "message": f"Cancelled job {job_id}"}).encode('utf-8'))
        else:
            self.wfile.write(json.dumps({"status": "error", "error": stderr}).encode('utf-8'))

    def handle_run_stream(self, query_string):
        global active_process
        
        params = urllib.parse.parse_qs(query_string)
        action = params.get('action', [''])[0]
        custom_cmd = params.get('cmd', [''])[0]
        
        # Setup Server-Sent Events headers
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        password = get_password()
        env = os.environ.copy()
        if password:
            env["SSHPASS"] = password

        # Determine target process command arguments
        args = []
        if action == "sync-code":
            # Push code excluding datasets and heavy objects
            src = DEFAULT_LOCAL_DIR
            dest = f"{CLUSTER_USER}@{CLUSTER_HOST}:{DEFAULT_REMOTE_DIR}"
            
            # Exclusions list
            exclusions = [
                ".git/", "data/", "outputs/", "**/__pycache__/", 
                "*.tflite", "*.task", "*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG",
                "diffvg/", "build/", "dist/", ".DS_Store", "scratch/"
            ]
            
            rsync_cmd = ["rsync", "-avz", "--progress"]
            for exc in exclusions:
                rsync_cmd.extend(["--exclude", exc])
                
            ssh_opts = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
            rsync_cmd.extend(["-e", ssh_opts, src, dest])
            
            if password:
                args = ["sshpass", "-e"] + rsync_cmd
            else:
                args = rsync_cmd
                
            self.send_sse_line(f"[LOCAL] Syncing Code: Local [{src}] ---> Cluster [{dest}]")

        elif action == "sync-data":
            # Push local dataset folder
            src = os.path.join(DEFAULT_LOCAL_DIR, "ControlSketch/data/")
            dest = f"{CLUSTER_USER}@{CLUSTER_HOST}:{DEFAULT_REMOTE_DIR}ControlSketch/data/"
            
            rsync_cmd = [
                "rsync", "-avz", "--progress",
                "-e", "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
                src, dest
            ]
            if password:
                args = ["sshpass", "-e"] + rsync_cmd
            else:
                args = rsync_cmd
                
            self.send_sse_line(f"[LOCAL] Syncing Datasets: Local [{src}] ---> Cluster [{dest}]")

        elif action == "sync-pull":
            # Pull cluster outputs to local outputs
            src = f"{CLUSTER_USER}@{CLUSTER_HOST}:{DEFAULT_REMOTE_DIR}outputs/"
            dest = os.path.join(DEFAULT_LOCAL_DIR, "outputs/")
            
            rsync_cmd = [
                "rsync", "-avz", "--progress",
                "-e", "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
                src, dest
            ]
            if password:
                args = ["sshpass", "-e"] + rsync_cmd
            else:
                args = rsync_cmd
                
            self.send_sse_line(f"[LOCAL] Syncing Results: Cluster [{src}] ---> Local [{dest}]")

        elif action == "remote-cmd":
            if not custom_cmd:
                self.send_sse_line("[ERROR] No remote command provided.")
                self.send_sse_finished(-1)
                return
                
            full_command = f"cd {DEFAULT_REMOTE_DIR} && {custom_cmd}"
            ssh_cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                f"{CLUSTER_USER}@{CLUSTER_HOST}",
                full_command
            ]
            if password:
                args = ["sshpass", "-e"] + ssh_cmd
            else:
                args = ssh_cmd
                
            self.send_sse_line(f"[CLUSTER] Executing: {custom_cmd}")
            
        else:
            self.send_sse_line(f"[ERROR] Unknown sync or execution action: {action}")
            self.send_sse_finished(-1)
            return

        try:
            active_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1
            )
            
            # Read stdout line-by-line in real-time
            for line in active_process.stdout:
                self.send_sse_line(line.rstrip('\r\n'))
                
            active_process.wait()
            code = active_process.returncode
            self.send_sse_finished(code)
        except Exception as e:
            self.send_sse_line(f"[ERROR] Subprocess error: {e}")
            self.send_sse_finished(-1)
        finally:
            active_process = None

    def send_sse_line(self, text):
        try:
            self.wfile.write(f"data: {text}\n\n".encode('utf-8'))
            self.wfile.flush()
        except Exception:
            pass

    def send_sse_finished(self, code):
        try:
            if code == 0:
                self.wfile.write(b"data: [PROCESS_COMPLETED]\n\n")
            else:
                self.wfile.write(f"data: [PROCESS_FAILED]: {code}\n\n".encode('utf-8'))
            self.wfile.flush()
        except Exception:
            pass

    def handle_abort(self):
        global active_process
        if active_process:
            print("[SERVER] Terminating active cluster interaction...")
            try:
                active_process.terminate()
                active_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                active_process.kill()
            active_process = None
            
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "aborted"}).encode('utf-8'))

def main():
    print("=" * 60)
    print("      SWIFTSKETCH PORTRAITRON SLURM PIPELINE MANAGER")
    print("=" * 60)
    print(f"Local Repository Path:  {DEFAULT_LOCAL_DIR}")
    print(f"Cluster Remote Path:    {DEFAULT_REMOTE_DIR}")
    print(f"Password Bypass File:   {PASSWORD_FILE} ({'FOUND' if os.path.exists(PASSWORD_FILE) else 'NOT FOUND'})")
    print("-" * 60)
    print(f"Server is starting at http://localhost:{PORT}")
    print("Open the above link in your web browser to orchestrate jobs.")
    print("Press Ctrl+C to terminate.")
    print("=" * 60)
    
    server = ThreadingHTTPServer(('0.0.0.0', PORT), SlurmHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server cleanly...")
        server.server_close()

if __name__ == '__main__':
    main()
