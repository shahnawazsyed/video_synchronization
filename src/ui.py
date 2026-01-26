"""
ui.py
------
Flask-based multi-step wizard UI for video synchronization.
"""
import os
import tempfile
import threading
import webbrowser
import zipfile
import logging
import logging.handlers
from flask import Flask, render_template_string, request, jsonify, send_from_directory, send_file

from .visual_sync import sync_videos_by_motion
from .video_sync import apply_video_offsets
from . import config

# Clean up any existing handlers to avoid duplication
root_logger = logging.getLogger()
if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

def configure_logging():
    """Setup centralized logging to file and console."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "video_sync.log")
    
    # Format: Time - LoggerName - Level - Message
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # 1. Rotating File Handler (10MB, 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # 2. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure Root Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info("Logging initialized. Writing to %s", log_file)

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "selected_files": [],
    "sync_progress": 0,
    "sync_status": "idle",
    "output_dir": tempfile.mkdtemp(prefix="sync_output_"),
    "current_step": 1,
    "offsets": {},
}

# --- HTML Templates ---

BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; color: #eee; }
.container { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
h1 { text-align: center; color: #00d9ff; margin-bottom: 8px; font-size: 28px; }
.subtitle { text-align: center; color: #888; margin-bottom: 40px; }
.step-indicator { display: flex; justify-content: center; margin-bottom: 40px; gap: 20px; }
.step-dot { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; background: rgba(255,255,255,0.1); color: #666; transition: all 0.3s; }
.step-dot.active { background: linear-gradient(135deg, #00d9ff, #0077ff); color: white; box-shadow: 0 0 20px rgba(0,217,255,0.5); }
.step-dot.done { background: #00cc66; color: white; }
.card { background: rgba(255,255,255,0.05); border-radius: 16px; padding: 30px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }
.btn { display: inline-block; padding: 14px 32px; border: none; border-radius: 10px; cursor: pointer; font-size: 15px; font-weight: 600; transition: all 0.3s; text-decoration: none; }
.btn-primary { background: linear-gradient(135deg, #00d9ff, #0077ff); color: white; }
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,217,255,0.4); }
.btn-primary:disabled { background: #444; cursor: not-allowed; transform: none; box-shadow: none; }
.btn-secondary { background: rgba(255,255,255,0.1); color: #aaa; }
.btn-secondary:hover { background: rgba(255,255,255,0.2); }
.btn-success { background: linear-gradient(135deg, #00cc66, #00aa55); color: white; }
.btn-success:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,204,102,0.4); }
.file-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 20px 0; }
.file-item { background: rgba(0,217,255,0.1); padding: 12px 16px; border-radius: 8px; font-size: 14px; }
.video-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 20px 0; }
.video-cell { background: #000; border-radius: 12px; overflow: hidden; aspect-ratio: 16/9; }
.video-cell video { width: 100%; height: 100%; object-fit: contain; }
.video-label { text-align: center; font-size: 12px; color: #888; margin-top: 5px; }
.progress-container { background: rgba(0,0,0,0.3); border-radius: 10px; height: 30px; overflow: hidden; margin: 20px 0; }
.progress-bar { height: 100%; background: linear-gradient(90deg, #00d9ff, #00cc66); width: 0%; transition: width 0.3s; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: bold; }
.status-text { text-align: center; color: #aaa; font-style: italic; margin: 15px 0; }
.btn-row { display: flex; justify-content: center; gap: 15px; margin-top: 30px; flex-wrap: wrap; }
.hidden { display: none; }
input[type="file"] { display: none; }
.select-btn-wrapper { text-align: center; margin: 30px 0; }
.seek-bar-container { margin: 20px 0; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); }
.seek-bar-label { text-align: center; font-size: 13px; color: #00d9ff; margin-bottom: 10px; font-weight: 600; }
.seek-bar { width: 100%; height: 8px; border-radius: 4px; background: rgba(255,255,255,0.2); outline: none; cursor: pointer; -webkit-appearance: none; appearance: none; }
.seek-bar::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; background: linear-gradient(135deg, #00d9ff, #0077ff); cursor: pointer; box-shadow: 0 0 10px rgba(0,217,255,0.5); }
.seek-bar::-moz-range-thumb { width: 18px; height: 18px; border-radius: 50%; background: linear-gradient(135deg, #00d9ff, #0077ff); cursor: pointer; border: none; box-shadow: 0 0 10px rgba(0,217,255,0.5); }
.seek-time { text-align: center; font-size: 12px; color: #888; margin-top: 8px; font-family: monospace; }
"""

STEP1_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Synchronization - Step 1</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <div class="container">
        <h1>Video Synchronization</h1>
        <p class="subtitle">Multi-camera synchronization wizard</p>
        
        <div class="step-indicator">
            <div class="step-dot active">1</div>
            <div class="step-dot">2</div>
            <div class="step-dot">3</div>
        </div>
        
        <div class="card">
            <h2 style="margin-bottom: 20px;">Step 1: Select Videos</h2>
            <p style="color: #888; margin-bottom: 20px;">Choose 2 or more video files</p>
            
            <div class="select-btn-wrapper">
                <label class="btn btn-primary" for="fileInput">Select Video Files</label>
                <input type="file" id="fileInput" multiple accept="video/*">
            </div>
            
            <div class="file-grid hidden" id="fileGrid"></div>
            
            <div class="btn-row">
                <button class="btn btn-primary" id="nextBtn" disabled onclick="goToStep2()">Continue to Preview</button>
            </div>
        </div>
    </div>
    
    <script>
        const fileInput = document.getElementById('fileInput');
        const fileGrid = document.getElementById('fileGrid');
        const nextBtn = document.getElementById('nextBtn');
        
        fileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            // Allow any number of files >= 2
            if (files.length < 2) {
                alert('Please select at least 2 video files.');
                return;
            }
            
            const formData = new FormData();
            files.forEach(f => formData.append('files[]', f));
            
            // Show loading state
            fileGrid.classList.remove('hidden');
            fileGrid.innerHTML = '<div class="status-text">Uploading files... please wait.</div>';
            
            fetch('/api/upload', {
                method: 'POST',
                body: formData
            }).then(r => r.json()).then(data => {
                if (data.ok) {
                    fileGrid.innerHTML = data.files.map(f => 
                        `<div class="file-item">${f}</div>`
                    ).join('');
                    nextBtn.disabled = false;
                } else {
                    alert(data.error);
                    fileGrid.innerHTML = '';
                    fileGrid.classList.add('hidden');
                }
            }).catch(err => {
                alert('Upload failed: ' + err);
                fileGrid.innerHTML = '';
                fileGrid.classList.add('hidden');
            });
        });
        
        function goToStep2() {
            window.location.href = '/step2';
        }
    </script>
</body>
</html>
"""

# HTML for Step 2: Preview
STEP2_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Step 2: Preview Videos</title>
    <style>
""" + BASE_CSS + """
    </style>
</head>
<body>
    <div class="container">
        <h1>Preview Original Videos</h1>
        <div class="subtitle">Check if videos are loaded correctly</div>
        
        <div class="step-indicator">
            <div class="step-dot done">1</div>
            <div class="step-dot active">2</div>
            <div class="step-dot">3</div>
        </div>
        
        <div class="card">
            <div class="video-grid">
                {% for file in files %}
                <div class="video-cell">
                    <video controls autoplay muted playsinline>
                        <source src="/video/raw/{{ file }}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <div class="video-label">{{ file }}</div>
                </div>
                {% endfor %}
            </div>
            
            <div class="btn-row">
                <button class="btn btn-secondary" onclick="window.history.back()">Back</button>
                <button class="btn btn-primary" onclick="startSync()">Start Synchronization</button>
            </div>
        </div>
    </div>
    
    <script>
        function startSync() {
            // Disable button and show progress
            const btn = document.querySelector('button.btn-primary');
            btn.disabled = true;
            btn.innerText = "Synchronizing...";
            
            fetch('/api/sync').then(r => r.json()).then(data => {
                if (data.ok) {
                    window.location.href = '/step3';
                } else {
                    alert(data.error);
                    btn.disabled = false;
                    btn.innerText = "Start Synchronization";
                }
            });
        }
    </script>
</body>
</html>
"""

# HTML for Step 3: Results
STEP3_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Step 3: Synchronization Results</title>
    <style>
""" + BASE_CSS + """
    </style>
</head>
<body>
    <div class="container">
        <h1>Synchronization Complete</h1>
        <div class="subtitle">Review synchronized playback</div>
        
        <div class="step-indicator">
            <div class="step-dot done">1</div>
            <div class="step-dot done">2</div>
            <div class="step-dot active">3</div>
        </div>
        
        <div class="card">
            <div id="progressSection">
                <div class="status-text" id="statusText">Processing...</div>
                <div class="progress-container">
                    <div class="progress-bar" id="progressBar">0%</div>
                </div>
            </div>
            
            <div id="resultSection" class="hidden">
                <div class="video-grid">
                    <!-- Synced videos will be injected here -->
                </div>
                
                <div class="btn-row">
                     <button id="playPauseBtn" class="btn btn-primary" onclick="togglePlayPause()">⏸ Pause</button>
                     <button class="btn btn-secondary" onclick="restartAllResult()">↺ Restart</button>
                </div>

                <div class="seek-bar-container">
                    <div class="seek-bar-label">Universal Seek Bar</div>
                    <input type="range" min="0" max="100" value="0" class="seek-bar" id="universalSeekBar">
                    <div class="seek-time" id="seekTimeDisplay">00:00 / 00:00</div>
                </div>
                
                <div class="btn-row">
                    <a href="/api/download_all" class="btn btn-success">Download All (ZIP)</a>
                    <a href="/" class="btn btn-secondary">Start Over</a>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let pollInterval;
        
        function checkProgress() {
            fetch('/api/progress').then(r => r.json()).then(data => {
                document.getElementById('statusText').innerText = data.status;
                document.getElementById('progressBar').style.width = data.progress + '%';
                document.getElementById('progressBar').innerText = data.progress + '%';
                
                if (data.progress >= 100) {
                    clearInterval(pollInterval);
                    setTimeout(showResults, 1000);
                }
            });
        }
        
        function showResults() {
            document.getElementById('progressSection').classList.add('hidden');
            
            fetch('/api/synced_files').then(r => r.json()).then(data => {
                if (data.ok) {
                    const grid = document.querySelector('#resultSection .video-grid');
                    grid.innerHTML = data.files.map(f => `
                        <div class="video-cell">
                            <video class="result-video" muted playsinline>
                                <source src="/video/synced/${f}" type="video/mp4">
                            </video>
                            <div class="video-label">${f}</div>
                        </div>
                    `).join('');
                    
                    document.getElementById('resultSection').classList.remove('hidden');
                    initSeekBar(); // Initialize the seek bar after videos are added
                    restartAllResult(); // Start playing automatically
                }
            });
        }
        
        function playAllResult() {
            document.querySelectorAll('.result-video').forEach(v => v.play());
            const btn = document.getElementById('playPauseBtn');
            btn.innerHTML = "⏸ Pause"; // Update button to 'Pause' state
        }
        
        function pauseAllResult() {
            document.querySelectorAll('.result-video').forEach(v => v.pause());
            const btn = document.getElementById('playPauseBtn');
            btn.innerHTML = "▶ Play"; // Update button to 'Play' state
        }
        
        function restartAllResult() {
            const videos = document.querySelectorAll('.result-video');
            const btn = document.getElementById('playPauseBtn');
            videos.forEach(v => {
                v.currentTime = 0;
                v.play();
            });
            if (btn) btn.innerHTML = "⏸ Pause";
        }
        
        function togglePlayPause() {
            const video = document.querySelector('.result-video');
            if (video && !video.paused) {
                 pauseAllResult();
            } else {
                 playAllResult();
            }
        }
        
        // --- Universal Seek Bar Logic ---
        const seekBar = document.getElementById('universalSeekBar');
        const timeDisplay = document.getElementById('seekTimeDisplay');
        let isDragging = false;
        
        function initSeekBar() {
            const videos = document.querySelectorAll('.result-video');
            if (videos.length === 0) return;
            
            // Assume all synced videos have same length (ensured by backend)
            // Wait for metadata to load to set max
            videos[0].addEventListener('loadedmetadata', () => {
                 seekBar.max = videos[0].duration;
                 updateTimeDisplay(0, videos[0].duration);
            });
            
            // Update slider as video plays
            setInterval(() => {
                if (!isDragging && !videos[0].paused) {
                    seekBar.value = videos[0].currentTime;
                    updateTimeDisplay(videos[0].currentTime, videos[0].duration);
                }
            }, 100);
            
            // Handle User Seek
            seekBar.addEventListener('input', () => {
                isDragging = true;
                const time = parseFloat(seekBar.value);
                videos.forEach(v => v.currentTime = time);
                updateTimeDisplay(time, videos[0].duration);
            });
            
            seekBar.addEventListener('change', () => {
                isDragging = false;
                const time = parseFloat(seekBar.value);
                videos.forEach(v => v.currentTime = time);
            });
        }
        
        function updateTimeDisplay(current, total) {
             const format = (t) => {
                 if (isNaN(t)) return "00:00";
                 const m = Math.floor(t / 60);
                 const s = Math.floor(t % 60);
                 return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
             };
             timeDisplay.innerText = `${format(current)} / ${format(total)}`;
        }
        
        pollInterval = setInterval(checkProgress, 1000);
    </script>
</body>
</html>
"""

# --- Routes ---

import uuid

@app.route('/')
def step1():
    app_state["current_step"] = 1
    app_state["selected_files"] = []
    app_state["sync_progress"] = 0
    app_state["session_id"] = f"req_{str(uuid.uuid4())[:8]}"
    return STEP1_HTML

@app.route('/step2')
def step2():
    if not app_state["selected_files"]:
        return '<script>window.location.href="/";</script>'
    app_state["current_step"] = 2
    from flask import render_template_string
    return render_template_string(STEP2_HTML, files=app_state["selected_files"])

@app.route('/step3')
def step3():
    if not app_state["selected_files"]:
        return '<script>window.location.href="/";</script>'
    app_state["current_step"] = 3
    return STEP3_HTML

@app.route('/api/select', methods=['POST'])
def api_select():
    sid = app_state.get("session_id", "unknown")
    data = request.json
    files = data.get('files', [])
    if len(files) < 2:
        logger.warning("[%s] Select failed: <2 files selected", sid)
        return jsonify({"ok": False, "error": "Please select at least 2 files."})
    
    # Validate files exist
    video_dir = config.VIDEO_DIR
    for fname in files:
        if not os.path.exists(os.path.join(video_dir, fname)):
            logger.error("[%s] Select failed: File not found %s", sid, fname)
            return jsonify({"ok": False, "error": f"File not found: {fname}. Make sure files are in {video_dir}"})
    
    app_state["selected_files"] = files
    logger.info("[%s] Selected files: %s", sid, files)
    return jsonify({"ok": True})

from . import preprocess
from . import audio_sync

@app.route('/api/sync')
def api_sync():
    sid = app_state.get("session_id", "unknown")
    if not app_state["selected_files"]:
        logger.warning("[%s] Sync failed: No files selected", sid)
        return jsonify({"ok": False, "error": "No files selected."})
    
    app_state["sync_progress"] = 0
    app_state["sync_status"] = "Initializing..."
    
    def run_sync():
        try:
            files = app_state["selected_files"]
            video_dir = config.VIDEO_DIR
            output_dir = app_state["output_dir"]
            
            method = getattr(config, "SYNC_METHOD", "visual")
            logger.info("[%s] Using synchronization method: %s", sid, method)
            
            if method == "audio":
                # Audio-based sync
                app_state["sync_progress"] = 10
                app_state["sync_status"] = "Extracting audio..."
                logger.info("[%s] Starting Audio Extraction...", sid)
                
                audio_dir = config.AUDIO_DIR
                preprocess.extract_audio_from_videos(video_dir, audio_dir)
                
                app_state["sync_progress"] = 40
                app_state["sync_status"] = "Analyzing audio offsets..."
                
                max_off = 20.0
                min_conf = 0.2
                logger.info("[%s] Estimating Audio Offsets (max_offset=%.1fs, min_confidence=%.2f)...", sid, max_off, min_conf)
                
                audio_offsets = audio_sync.estimate_offsets_robust(
                    audio_dir, 
                    max_offset_sec=max_off,
                    min_confidence=min_conf
                )
                
                # Convert wav filenames back to video filenames
                offsets = {}
                for vid_file in files:
                    wav_name = os.path.splitext(vid_file)[0] + ".wav"
                    if wav_name in audio_offsets:
                        offsets[vid_file] = audio_offsets[wav_name]
                    else:
                        logger.warning("[%s] No audio offset found for %s", sid, vid_file)
                        offsets[vid_file] = 0.0
                
            else:
                # Visual-based sync (default)
                app_state["sync_progress"] = 10
                app_state["sync_status"] = "Extracting motion energy..."
                logger.info("[%s] Starting Visual Sync...", sid)
                
                max_off = 20.0
                logger.info("[%s] Estimating Motion Offsets (max_offset=%.1fs)...", sid, max_off)
                
                offsets = sync_videos_by_motion(video_dir, files, max_offset_sec=max_off, output_dir=config.VISUAL_SYNC_OUTPUT_DIR)
            
            app_state["offsets"] = offsets
            logger.info("[%s] Offsets calculated: %s", sid, offsets)
            
            app_state["sync_progress"] = 60
            app_state["sync_status"] = "Applying offsets to videos..."
            
            apply_video_offsets(video_dir, offsets, output_dir)
            
            app_state["sync_progress"] = 100
            app_state["sync_status"] = "Complete!"
            logger.info("[%s] Synchronization process complete.", sid)
            
        except Exception as e:
            logger.error("[%s] Sync error: %s", sid, e, exc_info=True)
            app_state["sync_status"] = f"Error: {e}"
            app_state["sync_progress"] = 0
    
    thread = threading.Thread(target=run_sync, daemon=True)
    thread.start()
    
    return jsonify({"ok": True})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    sid = app_state.get("session_id", "unknown")
    if 'files[]' not in request.files:
        return jsonify({"ok": False, "error": "No files part in the request"})
    
    uploaded_files = request.files.getlist('files[]')
    
    if len(uploaded_files) < 2:
        return jsonify({"ok": False, "error": "Please upload at least 2 files"})
        
    saved_filenames = []
    
    # Ensure raw directory exists
    os.makedirs(config.VIDEO_DIR, exist_ok=True)
    
    from werkzeug.utils import secure_filename
    ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi'}
    
    for file in uploaded_files:
        if file.filename == '':
            continue
            
        _, ext = os.path.splitext(file.filename)
        if ext.lower() not in ALLOWED_EXTENSIONS:
            logger.warning("[%s] Upload rejected: Invalid file type %s", sid, file.filename)
            return jsonify({"ok": False, "error": f"File type not allowed: {file.filename}. Only .mp4, .mov, .avi allowed."})
            
        filename = secure_filename(file.filename)
        save_path = os.path.join(config.VIDEO_DIR, filename)
        
        try:
            file.save(save_path)
            saved_filenames.append(filename)
            logger.info("[%s] File uploaded: %s", sid, filename)
        except Exception as e:
            logger.error("Failed to save %s: %s", filename, e, exc_info=True)
            return jsonify({"ok": False, "error": f"Failed to save {filename}: {str(e)}"})
            
    app_state["selected_files"] = saved_filenames
    logger.info("Upload complete. Saved %d files.", len(saved_filenames))
    return jsonify({"ok": True, "files": saved_filenames})

@app.route('/api/progress')
def api_progress():
    return jsonify({
        "progress": app_state["sync_progress"],
        "status": app_state["sync_status"]
    })

@app.route('/api/synced_files')
def api_synced_files():
    files = app_state["selected_files"]
    output_dir = app_state["output_dir"]
    result = []
    for bname in files:
        base = os.path.splitext(bname)[0]
        for ext in ['.mp4', '.mov']:
            if os.path.exists(os.path.join(output_dir, f"{base}_synced{ext}")):
                result.append(f"{base}_synced{ext}")
                break
    return jsonify({"ok": True, "files": result})

@app.route('/api/download')
def api_download():
    """Create and send a ZIP file of all synced videos."""
    from io import BytesIO
    
    output_dir = app_state["output_dir"]
    files = app_state["selected_files"]
    
    # Create ZIP in memory using BytesIO
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for bname in files:
            base = os.path.splitext(bname)[0]
            for ext in ['.mp4', '.mov']:
                fpath = os.path.join(output_dir, f"{base}_synced{ext}")
                if os.path.exists(fpath):
                    zf.write(fpath, f"{base}_synced{ext}")
                    break
    
    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name="synced_videos.zip")

@app.route('/video/raw/<filename>')
def serve_raw_video(filename):
    return send_from_directory(os.path.abspath(config.VIDEO_DIR), filename)

@app.route('/video/synced/<filename>')
def serve_synced_video(filename):
    return send_from_directory(os.path.abspath(app_state["output_dir"]), filename)

def run_app():
    """Launch the web UI."""
    port = 5050
    url = f"http://127.0.0.1:{port}"
    print(f"\n{'='*50}")
    print(f"  Video Synchronization")
    print(f"  Open in browser: {url}")
    print(f"{'='*50}\n")
    
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False, threaded=True)
