import os
import sys
import json
import urllib.parse
import subprocess
import socketserver
import getpass
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8081  # Use 8081 to avoid conflict with the robot server on 8080
active_process = None
active_process_lock = threading.Lock()

# Automatically resolve paths relative to script location
SLURM_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCAL_DIR = os.path.dirname(SLURM_DIR) + "/"
DEFAULT_REMOTE_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/"
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

def log_to_file(text):
    log_path = os.path.join(SLURM_DIR, "slurm_web.log")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception as e:
        print(f"[WARNING] Failed to write log: {e}")

def generate_custom_slurm(job_name, remote_image_path, num_strokes, num_iter, feather_face_mask, condition, object_name, remote_project_dir):
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={remote_project_dir}/outputs/logs/{job_name}_%j.out
#SBATCH --error={remote_project_dir}/outputs/logs/{job_name}_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36000
#SBATCH --gpus=1

# 1. Activate environment
source ~/.bashrc
if ! command -v conda &> /dev/null; then
    source /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/anaconda3/bin/activate
fi

export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/huggingface_cache"
export CLIP_CACHE_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/clip_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda activate swiftsketch_env

# 2. Ensure log directory exists and navigate to project
mkdir -p {remote_project_dir}/outputs/logs
cd {remote_project_dir}

# 3. Run Object Sketching
python ControlSketch/object_sketching.py \\
  --target "{remote_image_path}" \\
  --num_strokes {num_strokes} \\
  --num_iter {num_iter} \\
  --feather_face_mask {feather_face_mask} \\
  --output_dir "{remote_project_dir}/outputs/{job_name}_run" \\
  --wandb_name "{job_name}" \\
  --condition "{condition}" \\
  --object_name "{object_name}"
"""

class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """
    Multi-threaded HTTP server using standard Python libraries.
    Allows concurrent requests, e.g. aborting sync/sbatch while streaming logs.
    """
    daemon_threads = True

    def handle_error(self, request, client_address):
        # Suppress harmless socket connection errors when client disconnects early (e.g. Broken Pipe)
        exc_type, exc_value, _ = sys.exc_info()
        if exc_type in (BrokenPipeError, ConnectionResetError):
            print(f"[INFO] Client {client_address} disconnected prematurely (Broken Pipe/Reset).")
            return
        super().handle_error(request, client_address)

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
        elif path == '/api/list-images':
            self.handle_list_images()
        elif path == '/api/view-image':
            self.handle_view_image(parsed_url.query)
        elif path == '/api/get-log':
            self.handle_get_log()
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
        elif path == '/api/clear-log':
            self.handle_clear_log()
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

    def handle_list_images(self):
        pictures_dir = os.path.join(DEFAULT_LOCAL_DIR, "Pictures")
        images = []
        if os.path.exists(pictures_dir):
            for root, _, files in os.walk(pictures_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.jpg_large')):
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, DEFAULT_LOCAL_DIR)
                        images.append({
                            "name": file,
                            "relative_path": rel_path,
                            "absolute_path": abs_path
                        })
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"images": images}).encode('utf-8'))

    def handle_view_image(self, query):
        params = urllib.parse.parse_qs(query)
        img_path = params.get('path', [''])[0]
        if not img_path or not os.path.exists(img_path):
            self.send_response(404)
            self.end_headers()
            return
            
        ext = os.path.splitext(img_path)[1].lower()
        mime_type = "image/png"
        if ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext == ".gif":
            mime_type = "image/gif"
        elif ext == ".svg":
            mime_type = "image/svg+xml"
            
        try:
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.end_headers()
            with open(img_path, "rb") as f:
                self.wfile.write(f.read())
        except Exception as e:
            try:
                self.send_response(500)
                self.end_headers()
            except:
                pass

    def handle_get_log(self):
        log_path = os.path.join(SLURM_DIR, "slurm_web.log")
        content = ""
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                content = f"Error reading log file: {e}"
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

    def handle_clear_log(self):
        log_path = os.path.join(SLURM_DIR, "slurm_web.log")
        try:
            if os.path.exists(log_path):
                os.remove(log_path)
            message = "Log file cleared successfully."
            status = "success"
        except Exception as e:
            message = f"Error clearing log file: {e}"
            status = "error"
            
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": status, "message": message}).encode('utf-8'))

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

    def run_stream_process(self, args, env, start_msg):
        global active_process
        self.send_sse_line(start_msg)
        try:
            with active_process_lock:
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
            return active_process.returncode
        except Exception as e:
            self.send_sse_line(f"[ERROR] Process failed to execute: {e}")
            return -1
        finally:
            with active_process_lock:
                active_process = None

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

        if action == "sync-code":
            # Check for dry run option
            dry_run = params.get('dry_run', ['false'])[0].lower() == 'true'
            
            # Push code excluding datasets and heavy objects
            src = DEFAULT_LOCAL_DIR
            dest = f"{CLUSTER_USER}@{CLUSTER_HOST}:{DEFAULT_REMOTE_DIR}SwiftSketch-Protraitron/"
            
            # Exclusions list
            exclusions = [
                ".git/", "data/", "outputs/", "**/__pycache__/", 
                "*.tflite", "*.task", "*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG",
                "diffvg/", "build/", "dist/", ".DS_Store", "scratch/"
            ]
            
            rsync_cmd = ["rsync", "-avz", "--progress"]
            if dry_run:
                rsync_cmd.append("--dry-run")
                
            for exc in exclusions:
                rsync_cmd.extend(["--exclude", exc])
                
            ssh_opts = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
            rsync_cmd.extend(["-e", ssh_opts, src, dest])
            
            if password:
                args = ["sshpass", "-e"] + rsync_cmd
            else:
                args = rsync_cmd
                
            start_msg = f"[LOCAL] (DRY RUN CHECK) Checking Code Sync: Local [{src}] ---> Cluster [{dest}]" if dry_run else f"[LOCAL] Syncing Code: Local [{src}] ---> Cluster [{dest}]"
            code = self.run_stream_process(args, env, start_msg)
            self.send_sse_finished(code)

        elif action == "sync-data":
            # Push local dataset folder
            src = os.path.join(DEFAULT_LOCAL_DIR, "ControlSketch/data/")
            dest = f"{CLUSTER_USER}@{CLUSTER_HOST}:{DEFAULT_REMOTE_DIR}SwiftSketch-Protraitron/ControlSketch/data/"
            
            rsync_cmd = [
                "rsync", "-avz", "--progress",
                "-e", "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
                src, dest
            ]
            if password:
                args = ["sshpass", "-e"] + rsync_cmd
            else:
                args = rsync_cmd
                
            code = self.run_stream_process(args, env, f"[LOCAL] Syncing Datasets: Local [{src}] ---> Cluster [{dest}]")
            self.send_sse_finished(code)

        elif action == "sync-pull":
            # Pull cluster outputs to local outputs
            src = f"{CLUSTER_USER}@{CLUSTER_HOST}:{DEFAULT_REMOTE_DIR}SwiftSketch-Protraitron/outputs/"
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
                
            code = self.run_stream_process(args, env, f"[LOCAL] Syncing Results: Cluster [{src}] ---> Local [{dest}]")
            self.send_sse_finished(code)

        elif action == "remote-cmd":
            if not custom_cmd:
                self.send_sse_line("[ERROR] No remote command provided.")
                self.send_sse_finished(-1)
                return
                
            full_command = f"cd {DEFAULT_REMOTE_DIR}SwiftSketch-Protraitron/ && {custom_cmd}"
            ssh_cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                f"{CLUSTER_USER}@{CLUSTER_HOST}",
                full_command
            ]
            if password:
                args = ["sshpass", "-e"] + ssh_cmd
            else:
                args = ssh_cmd
                
            code = self.run_stream_process(args, env, f"[CLUSTER] Executing: {custom_cmd}")
            self.send_sse_finished(code)

        elif action == "submit-single-job":
            local_image_path = params.get('image_path', [''])[0]
            num_strokes = params.get('num_strokes', ['98'])[0]
            num_iter = params.get('num_iter', ['1200'])[0]
            feather_face_mask = params.get('feather_face_mask', ['3'])[0]
            condition = params.get('condition', ['depth'])[0]
            object_name = params.get('object_name', ['face'])[0]
            suffix = params.get('suffix', [''])[0].strip()
            if 'suffix' not in params:
                print("[WARNING] 'suffix' query parameter was not provided in the API request. Defaulting to empty suffix.")
            
            if not local_image_path:
                self.send_sse_line("[ERROR] No target image path provided.")
                self.send_sse_finished(-1)
                return
                
            if not os.path.exists(local_image_path):
                self.send_sse_line(f"[ERROR] Local target image not found at path: {local_image_path}")
                self.send_sse_finished(-1)
                return
                
            basename = os.path.splitext(os.path.basename(local_image_path))[0]
            file_ext = os.path.splitext(local_image_path)[1]
            
            # Clean and append suffix if provided
            suffix_clean = re.sub(r'[^a-zA-Z0-9_\-]', '', suffix)
            if suffix_clean:
                job_name = f"ss_custom_{basename}_{num_strokes}s_{suffix_clean}"
            else:
                job_name = f"ss_custom_{basename}_{num_strokes}s"
            
            # Setup directories on cluster
            remote_images_dir = DEFAULT_REMOTE_DIR + "images"
            remote_jobs_dir = DEFAULT_REMOTE_DIR + "slurm_jobs"
            remote_project_dir = DEFAULT_REMOTE_DIR + "SwiftSketch-Protraitron"
            
            remote_image_path = f"{remote_images_dir}/{basename}{file_ext}"
            remote_slurm_path = f"{remote_jobs_dir}/{job_name}.slurm"
            
            # Step 1: Create remote dirs
            mkdir_cmd = f"mkdir -p {remote_images_dir} {remote_jobs_dir}"
            ssh_mkdir = [
                "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                f"{CLUSTER_USER}@{CLUSTER_HOST}",
                mkdir_cmd
            ]
            if password:
                args = ["sshpass", "-e"] + ssh_mkdir
            else:
                args = ssh_mkdir
                
            code = self.run_stream_process(args, env, f"[LOCAL] Creating remote directories on cluster: {remote_images_dir}, {remote_jobs_dir}")
            if code != 0:
                self.send_sse_line(f"[ERROR] Failed to create directories. Exit code: {code}")
                self.send_sse_finished(code)
                return
                
            # Step 2: Upload image
            rsync_img = [
                "rsync", "-avz", "--progress",
                "-e", "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
                local_image_path,
                f"{CLUSTER_USER}@{CLUSTER_HOST}:{remote_images_dir}/"
            ]
            if password:
                args = ["sshpass", "-e"] + rsync_img
            else:
                args = rsync_img
                
            code = self.run_stream_process(args, env, f"[LOCAL] Uploading target image to cluster: {local_image_path} ---> {remote_image_path}")
            if code != 0:
                self.send_sse_line(f"[ERROR] Failed to upload image. Exit code: {code}")
                self.send_sse_finished(code)
                return
                
            # Step 3: Generate local slurm script
            local_jobs_dir = os.path.join(SLURM_DIR, "jobs")
            os.makedirs(local_jobs_dir, exist_ok=True)
            local_slurm_path = os.path.join(local_jobs_dir, f"{job_name}.slurm")
            
            self.send_sse_line(f"[LOCAL] Generating custom SLURM script: {local_slurm_path}")
            try:
                slurm_content = generate_custom_slurm(
                    job_name=job_name,
                    remote_image_path=remote_image_path,
                    num_strokes=num_strokes,
                    num_iter=num_iter,
                    feather_face_mask=feather_face_mask,
                    condition=condition,
                    object_name=object_name,
                    remote_project_dir=remote_project_dir
                )
                with open(local_slurm_path, "w", encoding="utf-8") as f:
                    f.write(slurm_content)
            except Exception as e:
                self.send_sse_line(f"[ERROR] Failed to write local SLURM script: {e}")
                self.send_sse_finished(-1)
                return
                
            # Step 4: Upload slurm script
            rsync_slurm = [
                "rsync", "-avz", "--progress",
                "-e", "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
                local_slurm_path,
                f"{CLUSTER_USER}@{CLUSTER_HOST}:{remote_jobs_dir}/"
            ]
            if password:
                args = ["sshpass", "-e"] + rsync_slurm
            else:
                args = rsync_slurm
                
            code = self.run_stream_process(args, env, f"[LOCAL] Uploading SLURM script: {local_slurm_path} ---> {remote_slurm_path}")
            if code != 0:
                self.send_sse_line(f"[ERROR] Failed to upload script. Exit code: {code}")
                self.send_sse_finished(code)
                return
                
            # Step 5: Submit job
            sbatch_cmd = f"sbatch {remote_slurm_path}"
            ssh_sbatch = [
                "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                f"{CLUSTER_USER}@{CLUSTER_HOST}",
                sbatch_cmd
            ]
            if password:
                args = ["sshpass", "-e"] + ssh_sbatch
            else:
                args = ssh_sbatch
                
            code = self.run_stream_process(args, env, f"[CLUSTER] Submitting Slurm Job: {sbatch_cmd}")
            if code != 0:
                self.send_sse_line(f"[ERROR] Failed to submit Slurm Job. Exit code: {code}")
                self.send_sse_finished(code)
                return
                
            self.send_sse_line(f"[SUCCESS] Job {job_name} successfully submitted to Slurm cluster queue.")
            self.send_sse_finished(0)
            
        else:
            self.send_sse_line(f"[ERROR] Unknown sync or execution action: {action}")
            self.send_sse_finished(-1)

    def send_sse_line(self, text):
        log_to_file(text)
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
        with active_process_lock:
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
    
    server = ThreadingHTTPServer(('127.0.0.1', PORT), SlurmHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server cleanly...")
        server.server_close()

if __name__ == '__main__':
    main()
