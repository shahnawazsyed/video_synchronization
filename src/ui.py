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
from flask import Flask, render_template_string, request, jsonify, send_from_directory, send_file

from .visual_sync import sync_videos_by_motion
from .video_sync import apply_video_offsets
from . import config

app = Flask(__name__)

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
.video-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 20px 0; }
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
            <p style="color: #888; margin-bottom: 20px;">Choose exactly 4 video files from <code style="background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 4px;">data/raw/</code></p>
            
            <div class="select-btn-wrapper">
                <label class="btn btn-primary" for="fileInput">Select 4 Video Files</label>
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
            if (files.length !== 4) {
                alert('Please select exactly 4 video files.');
                return;
            }
            
            const fileNames = files.map(f => f.name);
            
            fetch('/api/select', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({files: fileNames})
            }).then(r => r.json()).then(data => {
                if (data.ok) {
                    fileGrid.classList.remove('hidden');
                    fileGrid.innerHTML = fileNames.map(f => 
                        `<div class="file-item">${f}</div>`
                    ).join('');
                    nextBtn.disabled = false;
                } else {
                    alert(data.error);
                }
            });
        });
        
        function goToStep2() {
            window.location.href = '/step2';
        }
    </script>
</body>
</html>
"""

STEP2_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Synchronization - Step 2</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <div class="container">
        <h1>Video Synchronization</h1>
        <p class="subtitle">Multi-camera synchronization wizard</p>
        
        <div class="step-indicator">
            <div class="step-dot done">1</div>
            <div class="step-dot active">2</div>
            <div class="step-dot">3</div>
        </div>
        
        <div class="card">
            <h2 style="margin-bottom: 20px;">Step 2: Preview (Before Sync)</h2>
            <p style="color: #888; margin-bottom: 20px;">Review your videos before synchronization. Notice the timing differences.</p>
            
            <div class="btn-row" style="margin-bottom: 20px;">
                <button class="btn btn-secondary" onclick="playAll()">Play All</button>
                <button class="btn btn-secondary" onclick="pauseAll()">Pause All</button>
                <button class="btn btn-secondary" onclick="restartAll()">Restart All</button>
            </div>
            
            <div class="video-grid" id="videoGrid">
                {% for file in files %}
                <div>
                    <div class="video-cell">
                        <video class="sync-video" controls muted loop>
                            <source src="/video/raw/{{ file }}" type="video/mp4">
                        </video>
                    </div>
                    <div class="video-label">{{ file }}</div>
                </div>
                {% endfor %}
            </div>
            
            <div class="btn-row">
                <button class="btn btn-secondary" onclick="goBack()">Back</button>
                <button class="btn btn-primary" onclick="startSync()">Start Synchronization</button>
            </div>
        </div>
    </div>
    
    <script>
        const videos = document.querySelectorAll('.sync-video');
        
        // Autoplay all videos when page loads
        window.addEventListener('load', () => {
            restartAll();
        });
        
        function playAll() {
            videos.forEach(v => v.play());
        }
        
        function pauseAll() {
            videos.forEach(v => v.pause());
        }
        
        function restartAll() {
            videos.forEach(v => {
                v.currentTime = 0;
                v.play();
            });
        }
        
        function goBack() { window.location.href = '/'; }
        function startSync() { window.location.href = '/step3'; }
    </script>
</body>
</html>
"""

STEP3_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Synchronization - Step 3</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <div class="container">
        <h1>Video Synchronization</h1>
        <p class="subtitle">Multi-camera synchronization wizard</p>
        
        <div class="step-indicator">
            <div class="step-dot done">1</div>
            <div class="step-dot done">2</div>
            <div class="step-dot active">3</div>
        </div>
        
        <div class="card">
            <h2 style="margin-bottom: 20px;">Step 3: Synchronization</h2>
            
            <div id="syncingView">
                <p style="color: #888; margin-bottom: 20px;">Processing your videos...</p>
                <div class="progress-container">
                    <div class="progress-bar" id="progressBar">0%</div>
                </div>
                <p class="status-text" id="statusText">Starting...</p>
            </div>
            
            <div id="resultView" class="hidden">
                <p style="color: #00cc66; margin-bottom: 20px;">Synchronization complete! Review the results below.</p>
                
                <div class="video-grid" id="resultGrid"></div>
                
                <!-- Playback Controls -->
                <div class="btn-row" style="margin: 20px 0 10px 0;">
                    <button class="btn btn-secondary" id="playPauseBtn" onclick="togglePlayPause()">▶ Play</button>
                    <button class="btn btn-secondary" onclick="restartAllResult()">⟲ Restart</button>
                </div>
                
                <!-- Universal Seek Bar -->
                <div class="seek-bar-container">
                    <div class="seek-bar-label">Universal Seek Control</div>
                    <input type="range" class="seek-bar" id="universalSeekBar" min="0" max="100" value="0" step="0.1">
                    <div class="seek-time" id="seekTime">0:00 / 0:00</div>
                </div>
                
                <div class="btn-row">
                    <button class="btn btn-secondary" onclick="location.href='/'">Start Over</button>
                    <a class="btn btn-success" href="/api/download" download>Download All Videos (ZIP)</a>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Start sync immediately when page loads
        fetch('/api/sync').then(r => r.json());
        
        // Poll progress
        const poll = setInterval(() => {
            fetch('/api/progress').then(r => r.json()).then(data => {
                document.getElementById('progressBar').style.width = data.progress + '%';
                document.getElementById('progressBar').textContent = data.progress + '%';
                document.getElementById('statusText').textContent = data.status;
                
                if (data.progress >= 100) {
                    clearInterval(poll);
                    setTimeout(showResults, 500);
                }
            });
        }, 500);
        
        function showResults() {
            document.getElementById('syncingView').classList.add('hidden');
            document.getElementById('resultView').classList.remove('hidden');
            
            // Load synced video grid
            fetch('/api/synced_files').then(r => r.json()).then(data => {
                const grid = document.getElementById('resultGrid');
                grid.innerHTML = data.files.map(f => `
                    <div>
                        <div class="video-cell">
                            <video class="result-video" controls muted loop>
                                <source src="/video/synced/${f}" type="video/mp4">
                            </video>
                        </div>
                        <div class="video-label">${f}</div>
                    </div>
                `).join('');
                
                // Initialize seek bar functionality and start playing
                setTimeout(() => {
                    initSeekBar();
                    restartAllResult(); // This will set button to "Pause" state
                }, 100);
            });
        }
        
        function initSeekBar() {
            const seekBar = document.getElementById('universalSeekBar');
            const seekTime = document.getElementById('seekTime');
            const videos = document.querySelectorAll('.result-video');
            
            if (!videos.length) return;
            
            let isUserSeeking = false;
            
            // Update seek bar when user drags it
            seekBar.addEventListener('input', (e) => {
                isUserSeeking = true;
                const percent = parseFloat(e.target.value);
                const firstVideo = videos[0];
                
                if (firstVideo.duration) {
                    const time = (percent / 100) * firstVideo.duration;
                    videos.forEach(v => v.currentTime = time);
                    updateTimeDisplay(time, firstVideo.duration);
                }
            });
            
            seekBar.addEventListener('change', () => {
                isUserSeeking = false;
            });
            
            // Update seek bar continuously as videos play
            setInterval(() => {
                if (!isUserSeeking && videos[0].duration) {
                    const currentTime = videos[0].currentTime;
                    const duration = videos[0].duration;
                    const percent = (currentTime / duration) * 100;
                    
                    seekBar.value = percent;
                    updateTimeDisplay(currentTime, duration);
                }
            }, 100);
            
            function updateTimeDisplay(current, total) {
                const formatTime = (seconds) => {
                    const mins = Math.floor(seconds / 60);
                    const secs = Math.floor(seconds % 60);
                    return `${mins}:${secs.toString().padStart(2, '0')}`;
                };
                
                seekTime.textContent = `${formatTime(current)} / ${formatTime(total)}`;
            }
        }
        
        function togglePlayPause() {
            const videos = document.querySelectorAll('.result-video');
            const btn = document.getElementById('playPauseBtn');
            
            if (!videos.length) return;
            
            // Check if videos are playing by checking the first video
            const isPlaying = !videos[0].paused;
            
            if (isPlaying) {
                videos.forEach(v => v.pause());
                btn.textContent = '▶ Play';
            } else {
                videos.forEach(v => v.play());
                btn.textContent = '⏸ Pause';
            }
        }
        
        function playAllResult() {
            const videos = document.querySelectorAll('.result-video');
            const btn = document.getElementById('playPauseBtn');
            videos.forEach(v => v.play());
            if (btn) btn.textContent = '⏸ Pause';
        }
        
        function pauseAllResult() {
            const videos = document.querySelectorAll('.result-video');
            const btn = document.getElementById('playPauseBtn');
            videos.forEach(v => v.pause());
            if (btn) btn.textContent = '▶ Play';
        }
        
        function restartAllResult() {
            const videos = document.querySelectorAll('.result-video');
            const btn = document.getElementById('playPauseBtn');
            videos.forEach(v => {
                v.currentTime = 0;
                v.play();
            });
            if (btn) btn.textContent = '⏸ Pause';
        }
    </script>
</body>
</html>
"""

# --- Routes ---

@app.route('/')
def step1():
    app_state["current_step"] = 1
    app_state["selected_files"] = []
    app_state["sync_progress"] = 0
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
    data = request.json
    files = data.get('files', [])
    if len(files) != 4:
        return jsonify({"ok": False, "error": "Please select exactly 4 files."})
    
    # Validate files exist
    video_dir = config.VIDEO_DIR
    for fname in files:
        if not os.path.exists(os.path.join(video_dir, fname)):
            return jsonify({"ok": False, "error": f"File not found: {fname}. Make sure files are in {video_dir}"})
    
    app_state["selected_files"] = files
    return jsonify({"ok": True})

from . import preprocess
from . import audio_sync

@app.route('/api/sync')
def api_sync():
    if not app_state["selected_files"]:
        return jsonify({"ok": False, "error": "No files selected."})
    
    app_state["sync_progress"] = 0
    app_state["sync_status"] = "Initializing..."
    
    def run_sync():
        try:
            files = app_state["selected_files"]
            video_dir = config.VIDEO_DIR
            output_dir = app_state["output_dir"]
            
            method = getattr(config, "SYNC_METHOD", "visual")
            print(f"Using synchronization method: {method}")
            
            if method == "audio":
                # Audio-based sync
                app_state["sync_progress"] = 10
                app_state["sync_status"] = "Extracting audio..."
                
                audio_dir = config.AUDIO_DIR
                preprocess.extract_audio_from_videos(video_dir, audio_dir)
                
                app_state["sync_progress"] = 40
                app_state["sync_status"] = "Analyzing audio offsets..."
                
                audio_offsets = audio_sync.estimate_offsets_robust(
                    audio_dir, 
                    max_offset_sec=20.0,
                    min_confidence=0.2
                )
                
                # Convert wav filenames back to video filenames
                offsets = {}
                for vid_file in files:
                    wav_name = os.path.splitext(vid_file)[0] + ".wav"
                    if wav_name in audio_offsets:
                        offsets[vid_file] = audio_offsets[wav_name]
                    else:
                        print(f"Warning: No audio offset found for {vid_file}")
                        offsets[vid_file] = 0.0
                
            else:
                # Visual-based sync (default)
                app_state["sync_progress"] = 10
                app_state["sync_status"] = "Extracting motion energy..."
                
                offsets = sync_videos_by_motion(video_dir, files, max_offset_sec=20.0, output_dir=config.VISUAL_SYNC_OUTPUT_DIR)
            
            app_state["offsets"] = offsets
            
            app_state["sync_progress"] = 60
            app_state["sync_status"] = "Applying offsets to videos..."
            
            apply_video_offsets(video_dir, offsets, output_dir)
            
            app_state["sync_progress"] = 100
            app_state["sync_status"] = "Complete!"
            
        except Exception as e:
            print(f"Sync error: {e}")
            app_state["sync_status"] = f"Error: {e}"
            app_state["sync_progress"] = 0
    
    thread = threading.Thread(target=run_sync, daemon=True)
    thread.start()
    
    return jsonify({"ok": True})

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
    return jsonify({"files": result})

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
