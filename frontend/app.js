// CourtVision AI - Advanced Tennis Biomechanics Controller

document.addEventListener('DOMContentLoaded', () => {
    // State management
    const state = {
        apiBase: (window.location.port === '3000' || window.location.port === '5500') ? 'http://127.0.0.1:8000' : window.location.origin,
        activeTab: 'analyze',
        statusCheckInterval: null,
        poseHistory: [],
        frameMetrics: [],
        shots: [],
        currentShotIndex: 0,
        currentFrame: 0,
        isPlaying: false,
        metricsChart: null,
        currentViewMode: 'canvas', // 'canvas' or 'annotated'
        annotatedVideoUrl: null
    };

    // DOM Elements
    const navButtons = document.querySelectorAll('.nav-item');
    const tabs = document.querySelectorAll('.tab-content');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const modelMode = document.getElementById('model-mode');
    const deviceInfo = document.getElementById('device-info');

    // Tab Navigation
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            switchTab(tabId);
        });
    });

    function switchTab(tabId) {
        state.activeTab = tabId;
        navButtons.forEach(btn => {
            if (btn.getAttribute('data-tab') === tabId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        tabs.forEach(tab => {
            if (tab.id === `tab-${tabId}`) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });

        const pageTitle = document.getElementById('page-title');
        const pageSubtitle = document.getElementById('page-subtitle');
        if (tabId === 'analyze') {
            pageTitle.innerText = "Biomechanics Dashboard";
            pageSubtitle.innerText = "Upload tennis videos or images to run pose estimation, stroke detection, and Transformer classification.";
        } else {
            pageTitle.innerText = "Help & Information";
            pageSubtitle.innerText = "Understand biomechanical metrics, joint calculations, and classification model details.";
        }
    }

    // Health/Status check
    async function checkServerStatus() {
        try {
            const res = await fetch(`${state.apiBase}/api/status`);
            if (res.ok) {
                const data = await res.json();
                statusDot.className = "status-dot online";
                statusText.innerText = "Server Online";
                modelMode.innerText = `Model: ${data.is_demo_mode ? 'Fallback Demo' : 'Trained Active'}`;
                if (deviceInfo) {
                    deviceInfo.innerText = `Device: ${data.device.toUpperCase()} (Dim: ${data.features.total_input_dim}x${data.features.sequence_len})`;
                }
            } else {
                throw new Error("Offline response");
            }
        } catch (err) {
            statusDot.className = "status-dot";
            statusText.innerText = "Server Offline";
            modelMode.innerText = "Mode: Disconnected";
            if (deviceInfo) {
                deviceInfo.innerText = "Device: N/A";
            }
        }
    }

    checkServerStatus();
    state.statusCheckInterval = setInterval(checkServerStatus, 5000);

    // --- Tab 1: Analyze Stroke Elements ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const selectFileBtn = document.getElementById('select-file-btn');
    const loadSampleVideoBtn = document.getElementById('load-sample-video-btn');
    const processingCard = document.getElementById('processing-card');
    const visualizerCard = document.getElementById('visualizer-card');
    const annotatedImageView = document.getElementById('annotated-image-view');
    const videoWrapper = document.getElementById('video-wrapper');
    const sourceVideoView = document.getElementById('source-video-view');
    const annotatedVideoView = document.getElementById('annotated-video-view');
    const skeletonCanvas = document.getElementById('skeleton-canvas');
    const ctx = skeletonCanvas.getContext('2d');
    
    const viewModeGroup = document.getElementById('view-mode-group');
    const btnModeCanvas = document.getElementById('btn-mode-canvas');
    const btnModeAnnotated = document.getElementById('btn-mode-annotated');
    const downloadVideoBtn = document.getElementById('download-video-btn');

    const playPauseBtn = document.getElementById('play-pause-btn');
    const timelineScrubber = document.getElementById('timeline-scrubber');
    const frameCounter = document.getElementById('frame-counter');
    const playbackControls = document.getElementById('playback-controls');

    const insightsEmptyState = document.getElementById('insights-empty-state');
    const insightsContent = document.getElementById('insights-content');
    const valStroke = document.getElementById('val-stroke');
    const valStrokeConf = document.getElementById('val-stroke-conf');
    const confStrokeBar = document.getElementById('conf-stroke-bar');
    const probStrokeMini = document.getElementById('prob-stroke-mini');

    const valDirection = document.getElementById('val-direction');
    const valDirectionConf = document.getElementById('val-direction-conf');
    const confDirectionBar = document.getElementById('conf-direction-bar');
    const probDirectionMini = document.getElementById('prob-direction-mini');

    const valPosture = document.getElementById('val-posture');
    const valPostureConf = document.getElementById('val-posture-conf');
    const confPostureBar = document.getElementById('conf-posture-bar');
    const probPostureMini = document.getElementById('prob-posture-mini');

    const recTitle = document.getElementById('rec-title');
    const recText = document.getElementById('rec-text');
    
    const anglesEmptyState = document.getElementById('angles-empty-state');
    const chartContainer = document.getElementById('chart-container');
    const reuploadBtn = document.getElementById('reupload-btn');
    const playbackImpactFrame = document.getElementById('playback-impact-frame');

    // Multi-Shot Navigator Elements
    const shotNavigatorCard = document.getElementById('shot-navigator-card');
    const prevShotBtn = document.getElementById('prev-shot-btn');
    const nextShotBtn = document.getElementById('next-shot-btn');
    const currentShotBadge = document.getElementById('current-shot-badge');
    const shotTimestampTag = document.getElementById('shot-timestamp-tag');
    const shotPillsBar = document.getElementById('shot-pills-bar');
    const loadCoricVideoBtn = document.getElementById('load-coric-video-btn');
    const insightsCardTitle = document.getElementById('insights-card-title');

    // Trigger file chooser
    selectFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Borna Coric Rally Video Runner (Reference multi-shot match video)
    if (loadCoricVideoBtn) {
        loadCoricVideoBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
                dropZone.classList.add('hidden');
                processingCard.classList.remove('hidden');
                const uploadProgressText = document.getElementById('upload-progress-text');
                if (uploadProgressText) uploadProgressText.innerText = "Loading Borna Coric match rally video from server...";
                
                const res = await fetch(`${state.apiBase}/demo/coric_tennis.mp4`);
                if (!res.ok) throw new Error("Could not load Borna Coric demo video");
                const blob = await res.blob();
                const demoFile = new File([blob], "coric_tennis.mp4", { type: "video/mp4" });
                handleUploadedFile(demoFile);
            } catch (err) {
                alert("Failed to load Coric video: " + err.message);
                processingCard.classList.add('hidden');
                dropZone.classList.remove('hidden');
            }
        });
    }

    // Sample video runner (Dominic Thiem)
    if (loadSampleVideoBtn) {
        loadSampleVideoBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
                dropZone.classList.add('hidden');
                processingCard.classList.remove('hidden');
                const uploadProgressText = document.getElementById('upload-progress-text');
                if (uploadProgressText) uploadProgressText.innerText = "Loading Dominic Thiem sample video from server...";
                
                const res = await fetch(`${state.apiBase}/demo/sample_tennis.mp4`);
                if (!res.ok) throw new Error("Could not load demo video");
                const blob = await res.blob();
                const demoFile = new File([blob], "sample_tennis.mp4", { type: "video/mp4" });
                handleUploadedFile(demoFile);
            } catch (err) {
                alert("Failed to load sample video: " + err.message);
                processingCard.classList.add('hidden');
                dropZone.classList.remove('hidden');
            }
        });
    }

    if (reuploadBtn) {
        reuploadBtn.addEventListener('click', () => {
            resetVisualizer();
            dropZone.classList.remove('hidden');
        });
    }

    // Drag-and-drop listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleUploadedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleUploadedFile(fileInput.files[0]);
        }
    });

    // View Mode Toggle (Canvas vs Pre-rendered Annotated Video)
    if (btnModeCanvas && btnModeAnnotated) {
        btnModeCanvas.addEventListener('click', () => {
            switchViewMode('canvas');
        });
        btnModeAnnotated.addEventListener('click', () => {
            switchViewMode('annotated');
        });
    }

    function switchViewMode(mode) {
        state.currentViewMode = mode;
        if (mode === 'canvas') {
            btnModeCanvas.classList.add('active');
            btnModeCanvas.style.background = 'var(--primary-color)';
            btnModeCanvas.style.color = '#080c14';

            btnModeAnnotated.classList.remove('active');
            btnModeAnnotated.style.background = 'transparent';
            btnModeAnnotated.style.color = 'var(--text-secondary)';

            videoWrapper.classList.remove('hidden');
            annotatedVideoView.classList.add('hidden');
            playbackControls.classList.remove('hidden');
            annotatedVideoView.pause();
            sourceVideoView.play().catch(() => {});
        } else {
            btnModeAnnotated.classList.add('active');
            btnModeAnnotated.style.background = 'var(--primary-color)';
            btnModeAnnotated.style.color = '#080c14';

            btnModeCanvas.classList.remove('active');
            btnModeCanvas.style.background = 'transparent';
            btnModeCanvas.style.color = 'var(--text-secondary)';

            videoWrapper.classList.add('hidden');
            annotatedVideoView.classList.remove('hidden');
            playbackControls.classList.add('hidden');
            sourceVideoView.pause();
            annotatedVideoView.play().catch(() => {});
        }
    }

    function handleUploadedFile(file) {
        resetVisualizer();

        const isImage = file.type.startsWith('image/');
        const isVideo = file.type.startsWith('video/') || file.name.endsWith('.mp4') || file.name.endsWith('.mov');

        if (!isImage && !isVideo) {
            alert('Unsupported file format. Please upload an image or video.');
            return;
        }

        const maxSizeBytes = 150 * 1024 * 1024;
        if (file.size > maxSizeBytes) {
            alert(`File size exceeds 150MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB). Please choose a smaller file.`);
            return;
        }

        // Show processing indicator
        dropZone.classList.add('hidden');
        processingCard.classList.remove('hidden');

        const uploadProgressBar = document.getElementById('upload-progress-bar');
        const uploadProgressText = document.getElementById('upload-progress-text');
        
        if (uploadProgressBar) uploadProgressBar.style.width = '0%';
        if (uploadProgressText) uploadProgressText.innerText = 'Uploading: 0% – Calculating ETA...';

        const formData = new FormData();
        formData.append('file', file);

        const startTime = Date.now();
        const xhr = new XMLHttpRequest();
        const endpoint = isImage ? '/api/analyze-image' : '/api/analyze-video';

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable && e.total > 0) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                const elapsedTime = (Date.now() - startTime) / 1000;
                const speed = elapsedTime > 0 ? (e.loaded / elapsedTime) : 0;
                const remainingBytes = e.total - e.loaded;
                const etaSeconds = speed > 0 ? Math.ceil(remainingBytes / speed) : 0;

                const minutes = Math.floor(etaSeconds / 60);
                const seconds = etaSeconds % 60;
                const etaFormatted = minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;

                if (uploadProgressBar) uploadProgressBar.style.width = `${percentComplete}%`;
                if (uploadProgressText) {
                    if (percentComplete < 100) {
                        uploadProgressText.innerText = `Upload progress: ${percentComplete}% – ETA: ${etaFormatted}`;
                    } else {
                        uploadProgressText.innerText = 'Upload complete! Running MediaPipe pose detection & Transformer inference...';
                    }
                }
            }
        };

        xhr.onload = () => {
            processingCard.classList.add('hidden');
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const data = JSON.parse(xhr.responseText);
                    if (data.success) {
                        displayAnalysisResults(data, file, isImage);
                    } else {
                        alert(`Analysis error: ${data.message}`);
                        dropZone.classList.remove('hidden');
                    }
                } catch (e) {
                    alert('Error parsing response from server: ' + e.message);
                    dropZone.classList.remove('hidden');
                }
            } else {
                let detail = 'Analysis failed';
                try {
                    const err = JSON.parse(xhr.responseText);
                    detail = err.detail || detail;
                } catch (errObj) {}
                alert(`Error (${xhr.status}): ${detail}`);
                dropZone.classList.remove('hidden');
            }
        };

        xhr.onerror = () => {
            alert('Network error communicating with the server.');
            processingCard.classList.add('hidden');
            dropZone.classList.remove('hidden');
        };

        xhr.open('POST', `${state.apiBase}${endpoint}`, true);
        xhr.send(formData);
    }

    const videoFeedbackBar = document.getElementById('video-feedback-bar');
    const playbackTimeDisplay = document.getElementById('playback-time-display');
    const playbackBufferingStatus = document.getElementById('playback-buffering-status');
    const playbackSpeedSelector = document.getElementById('playback-speed-selector');

    function formatTime(seconds) {
        if (isNaN(seconds) || seconds === Infinity) return "00:00";
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }

    if (sourceVideoView) {
        sourceVideoView.addEventListener('timeupdate', () => {
            const current = formatTime(sourceVideoView.currentTime);
            const total = formatTime(sourceVideoView.duration);
            if (playbackTimeDisplay) {
                playbackTimeDisplay.innerText = `${current} / ${total}`;
            }

            // High-precision shot synchronization during playback
            if (state.shots && state.shots.length > 0 && !sourceVideoView.paused) {
                const curTime = sourceVideoView.currentTime;
                const activeIdx = state.shots.findIndex(s => {
                    const start = (s.start_time !== undefined) ? s.start_time : (s.peak_time - 0.5);
                    const end = (s.end_time !== undefined) ? s.end_time : (s.peak_time + 0.5);
                    return curTime >= (start - 0.15) && curTime <= (end + 0.15);
                });

                if (activeIdx !== -1 && activeIdx !== state.currentShotIndex) {
                    selectShot(activeIdx, false); // Don't seek video, playback is continuous!
                }
            }
        });

        sourceVideoView.addEventListener('waiting', () => {
            if (playbackBufferingStatus) {
                playbackBufferingStatus.innerHTML = '<span class="status-badge buffering">Buffering...</span>';
            }
        });

        sourceVideoView.addEventListener('playing', () => {
            if (playbackBufferingStatus) {
                playbackBufferingStatus.innerHTML = '<span class="status-badge ready">Ready</span>';
            }
        });

        sourceVideoView.addEventListener('canplay', () => {
            if (playbackBufferingStatus) {
                playbackBufferingStatus.innerHTML = '<span class="status-badge ready">Ready</span>';
            }
        });
    }

    if (playbackSpeedSelector) {
        playbackSpeedSelector.addEventListener('change', () => {
            const rate = parseFloat(playbackSpeedSelector.value);
            if (sourceVideoView) sourceVideoView.playbackRate = rate;
            if (annotatedVideoView) annotatedVideoView.playbackRate = rate;
        });
    }

    function resetVisualizer() {
        pauseSkeleton();
        state.poseHistory = [];
        state.frameMetrics = [];
        state.shots = [];
        state.currentShotIndex = 0;
        state.currentFrame = 0;
        state.annotatedVideoUrl = null;
        
        if (shotNavigatorCard) shotNavigatorCard.classList.add('hidden');
        if (shotPillsBar) shotPillsBar.innerHTML = '';
        if (playbackImpactFrame) playbackImpactFrame.innerText = "Shot 1";

        annotatedImageView.classList.add('hidden');
        videoWrapper.classList.add('hidden');
        annotatedVideoView.classList.add('hidden');
        viewModeGroup.classList.add('hidden');
        downloadVideoBtn.classList.add('hidden');
        playbackControls.classList.add('hidden');
        if (videoFeedbackBar) videoFeedbackBar.classList.add('hidden');
        visualizerCard.classList.add('hidden');
        
        insightsContent.classList.add('hidden');
        insightsEmptyState.classList.remove('hidden');
        chartContainer.classList.add('hidden');
        anglesEmptyState.classList.remove('hidden');

        probStrokeMini.innerHTML = '';
        probDirectionMini.innerHTML = '';
        probPostureMini.innerHTML = '';
    }

    function displayAnalysisResults(data, file, isImage) {
        visualizerCard.classList.remove('hidden');

        if (isImage) {
            if (shotNavigatorCard) shotNavigatorCard.classList.add('hidden');
            annotatedImageView.src = data.image_data;
            annotatedImageView.classList.remove('hidden');
            document.getElementById('visualizer-title').innerText = "Static Pose Analysis";
            
            insightsEmptyState.classList.add('hidden');
            insightsContent.classList.remove('hidden');
            if (insightsCardTitle) insightsCardTitle.innerText = "Stroke Analysis";
            
            valStroke.innerText = formatLabel(data.stroke_type || "Setup");
            valStrokeConf.innerText = data.stroke_confidence ? `${Math.round(data.stroke_confidence * 100)}% Conf` : "Static";
            confStrokeBar.style.width = data.stroke_confidence ? `${Math.round(data.stroke_confidence * 100)}%` : "100%";
            
            valDirection.innerText = formatLabel(data.direction || "Center");
            valDirectionConf.innerText = data.direction_confidence ? `${Math.round(data.direction_confidence * 100)}% Conf` : "Static";
            confDirectionBar.style.width = data.direction_confidence ? `${Math.round(data.direction_confidence * 100)}%` : "100%";
            
            valPosture.innerText = formatLabel(data.posture || "Analyzed");
            valPostureConf.innerText = data.posture_confidence ? `${Math.round(data.posture_confidence * 100)}% Conf` : "Static";
            confPostureBar.style.width = data.posture_confidence ? `${Math.round(data.posture_confidence * 100)}%` : "100%";

            if ((data.posture || '').toLowerCase() === 'bad') {
                valPosture.style.color = 'var(--danger-color)';
            } else {
                valPosture.style.color = 'var(--primary-color)';
            }

            recTitle.innerText = data.recommendation_title || "Pose Evaluated";
            recText.innerHTML = data.recommendation_text || "Pose extracted and angles evaluated.";

            renderAnglesChart(data.metrics, true);
        } else {
            // Video analysis
            document.getElementById('visualizer-title').innerText = "Stroke Biomechanics Playback";
            videoWrapper.classList.remove('hidden');
            playbackControls.classList.remove('hidden');
            if (videoFeedbackBar) videoFeedbackBar.classList.remove('hidden');

            // Interactive local video player
            sourceVideoView.src = URL.createObjectURL(file);
            sourceVideoView.muted = true;
            sourceVideoView.loop = true;
            sourceVideoView.play().catch(e => console.log("Autoplay blocked", e));

            // Pre-rendered annotated video setup
            if (data.annotated_video_url) {
                state.annotatedVideoUrl = data.annotated_video_url;
                annotatedVideoView.src = `${state.apiBase}${data.annotated_video_url}`;
                annotatedVideoView.loop = true;
                
                viewModeGroup.classList.remove('hidden');
                downloadVideoBtn.href = `${state.apiBase}${data.annotated_video_url}`;
                downloadVideoBtn.classList.remove('hidden');
            }

            // Pose history & scrubber
            state.poseHistory = data.pose_history || [];
            state.frameMetrics = data.frame_metrics || [];
            state.shots = data.shots || [];
            timelineScrubber.max = Math.max(0, state.poseHistory.length - 1);
            timelineScrubber.value = 0;
            updateFrameCounterText();

            // Insights Content unhide
            insightsEmptyState.classList.add('hidden');
            insightsContent.classList.remove('hidden');

            // Populate multi-shot navigator
            if (state.shots.length > 0) {
                if (shotNavigatorCard) shotNavigatorCard.classList.remove('hidden');
                if (shotPillsBar) {
                    shotPillsBar.innerHTML = '';
                    state.shots.forEach((shot, idx) => {
                        const pill = document.createElement('button');
                        pill.type = 'button';
                        pill.className = `shot-pill ${idx === 0 ? 'active' : ''}`;
                        pill.id = `shot-pill-${idx}`;
                        const strokeName = formatLabel(shot.stroke_type || 'Shot');
                        pill.innerHTML = `<span>#${shot.shot_id || idx + 1}</span> <span class="shot-pill-type">${strokeName}</span>`;
                        pill.addEventListener('click', (e) => {
                            e.stopPropagation();
                            selectShot(idx, true);
                        });
                        shotPillsBar.appendChild(pill);
                    });
                }

                // Initialize display with Shot 1 (index 0)
                selectShot(0, false);
            } else {
                // Fallback for videos with single or no detected stroke peaks
                if (shotNavigatorCard) shotNavigatorCard.classList.add('hidden');
                valStroke.innerText = formatLabel(data.stroke_type || 'Rally');
                const strokeBasis = data.stroke_basis ? ` • ${data.stroke_basis}` : '';
                valStrokeConf.innerText = data.stroke_confidence ? `${Math.round(data.stroke_confidence * 100)}% Conf${strokeBasis}` : '100%';
                confStrokeBar.style.width = data.stroke_confidence ? `${Math.round(data.stroke_confidence * 100)}%` : '100%';
                renderProbList(probStrokeMini, data.stroke_probabilities);

                valDirection.innerText = formatLabel(data.direction || 'Center');
                const dirBasis = data.direction_basis ? ` • ${data.direction_basis}` : '';
                valDirectionConf.innerText = data.direction_confidence ? `${Math.round(data.direction_confidence * 100)}% Conf${dirBasis}` : '100%';
                confDirectionBar.style.width = data.direction_confidence ? `${Math.round(data.direction_confidence * 100)}%` : '100%';
                renderProbList(probDirectionMini, data.direction_probabilities);

                const pScore = data.posture_score !== undefined ? data.posture_score : (data.posture_quality?.score ?? 75);
                const pGrade = data.posture_grade || (data.posture_quality?.grade ?? (data.posture === 'good' ? 'Good' : 'Needs Improvement'));
                valPosture.innerText = `${pGrade} (${pScore}/100)`;
                valPostureConf.innerText = data.posture_confidence ? `${Math.round(data.posture_confidence * 100)}% Biomechanical Score` : '75% Biomechanical Score';
                confPostureBar.style.width = `${Math.min(100, Math.max(0, pScore))}%`;
                if (pScore >= 82) {
                    valPosture.style.color = 'var(--primary-color)';
                    confPostureBar.style.background = 'var(--primary-color)';
                } else if (pScore >= 68) {
                    valPosture.style.color = '#38bdf8';
                    confPostureBar.style.background = '#38bdf8';
                } else {
                    valPosture.style.color = 'var(--danger-color)';
                    confPostureBar.style.background = 'var(--danger-color)';
                }
                renderProbList(probPostureMini, data.posture_probabilities);

                recTitle.innerText = data.recommendation_title || 'Pose Extracted';
                recText.innerHTML = data.recommendation_text || '';

                renderAnglesChart(data.metrics, false);
            }

            // Trigger skeleton playback
            startSkeletonPlay();
        }
    }

    function selectShot(index, seekVideo = true) {
        if (!state.shots || state.shots.length === 0) return;
        if (index < 0 || index >= state.shots.length) return;
        
        state.currentShotIndex = index;
        const shot = state.shots[index];
        const total = state.shots.length;

        // Update navigator UI badge and timestamps
        if (currentShotBadge) {
            currentShotBadge.innerText = `Shot ${index + 1} of ${total}`;
        }
        if (shotTimestampTag) {
            const startStr = formatTime(shot.start_time);
            const endStr = formatTime(shot.end_time);
            const peakStr = formatTime(shot.peak_time);
            shotTimestampTag.innerText = `⏱ ${startStr} - ${endStr} (Impact: ${peakStr})`;
        }
        if (prevShotBtn) {
            prevShotBtn.disabled = (index === 0);
            prevShotBtn.style.opacity = (index === 0) ? '0.4' : '1';
        }
        if (nextShotBtn) {
            nextShotBtn.disabled = (index === total - 1);
            nextShotBtn.style.opacity = (index === total - 1) ? '0.4' : '1';
        }

        // Update pill active classes
        if (shotPillsBar) {
            const pills = shotPillsBar.querySelectorAll('.shot-pill');
            pills.forEach((p, idx) => {
                if (idx === index) {
                    p.classList.add('active');
                    p.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
                } else {
                    p.classList.remove('active');
                }
            });
        }

        // Update feedback bar active shot
        if (playbackImpactFrame) {
            playbackImpactFrame.innerText = `Shot ${index + 1}: ${formatLabel(shot.stroke_type)} @ ${formatTime(shot.peak_time)}`;
        }

        // Update Insights card title
        if (insightsCardTitle) {
            insightsCardTitle.innerText = `Shot #${index + 1} Analysis (${formatLabel(shot.stroke_type)})`;
        }

        // Update Stroke Type & confidence
        valStroke.innerText = formatLabel(shot.stroke_type);
        const strokeBasisText = shot.stroke_basis ? ` • ${shot.stroke_basis}` : '';
        valStrokeConf.innerText = `${Math.round(shot.stroke_confidence * 100)}% Conf${strokeBasisText}`;
        confStrokeBar.style.width = `${Math.round(shot.stroke_confidence * 100)}%`;
        renderProbList(probStrokeMini, shot.stroke_probabilities);

        // Update Direction & confidence
        valDirection.innerText = formatLabel(shot.direction);
        const dirBasisText = shot.direction_basis ? ` • ${shot.direction_basis}` : '';
        valDirectionConf.innerText = `${Math.round(shot.direction_confidence * 100)}% Conf${dirBasisText}`;
        confDirectionBar.style.width = `${Math.round(shot.direction_confidence * 100)}%`;
        renderProbList(probDirectionMini, shot.direction_probabilities);

        // Update Posture & confidence with 0-100 multi-factor score
        const pScore = shot.posture_score !== undefined ? shot.posture_score : (shot.posture_quality?.score ?? 75);
        const pGrade = shot.posture_grade || (shot.posture_quality?.grade ?? (shot.posture === 'good' ? 'Good' : 'Needs Improvement'));
        valPosture.innerText = `${pGrade} (${pScore}/100)`;
        valPostureConf.innerText = `${Math.round((shot.posture_confidence || (pScore/100)) * 100)}% Biomechanical Score`;
        confPostureBar.style.width = `${Math.min(100, Math.max(0, pScore))}%`;
        if (pScore >= 82) {
            valPosture.style.color = 'var(--primary-color)';
            confPostureBar.style.background = 'var(--primary-color)';
        } else if (pScore >= 68) {
            valPosture.style.color = '#38bdf8';
            confPostureBar.style.background = '#38bdf8';
        } else {
            valPosture.style.color = 'var(--danger-color)';
            confPostureBar.style.background = 'var(--danger-color)';
        }
        renderProbList(probPostureMini, shot.posture_probabilities);

        // Update Recommendation for this specific shot
        recTitle.innerText = shot.recommendation_title || "Shot Analysis";
        recText.innerHTML = shot.recommendation_text || "";

        // DYNAMIC BIOMECHANICAL ANGLES UPDATE
        // shot.metrics contains the dynamic joint angles specifically aggregated for this shot!
        renderAnglesChart(shot.metrics, false);

        // Synchronize Video Playhead if user clicked navigation
        if (seekVideo && sourceVideoView && sourceVideoView.duration > 0) {
            const targetTime = shot.start_time !== undefined ? shot.start_time : shot.peak_time;
            sourceVideoView.currentTime = Math.max(0, targetTime);
            
            // Also sync timeline scrubber and skeleton
            if (state.poseHistory.length > 0 && shot.peak_frame !== undefined) {
                state.currentFrame = Math.min(state.poseHistory.length - 1, shot.peak_frame);
                timelineScrubber.value = state.currentFrame;
                updateFrameCounterText();
                drawSkeletonFrame(state.currentFrame);
            }
        }
    }

    if (prevShotBtn) {
        prevShotBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (state.currentShotIndex > 0) {
                selectShot(state.currentShotIndex - 1, true);
            }
        });
    }
    if (nextShotBtn) {
        nextShotBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (state.currentShotIndex < state.shots.length - 1) {
                selectShot(state.currentShotIndex + 1, true);
            }
        });
    }

    function renderProbList(container, probMap) {
        if (!container || !probMap) return;
        container.innerHTML = '';
        const wrapper = document.createElement('div');
        wrapper.style.display = 'flex';
        wrapper.style.gap = '4px';
        wrapper.style.flexWrap = 'wrap';
        wrapper.style.marginTop = '4px';

        for (const [cls, prob] of Object.entries(probMap)) {
            const badge = document.createElement('span');
            badge.style.fontSize = '0.65rem';
            badge.style.padding = '1px 5px';
            badge.style.borderRadius = '3px';
            badge.style.background = 'rgba(255,255,255,0.05)';
            badge.style.color = 'var(--text-secondary)';
            badge.innerText = `${formatLabel(cls)}: ${Math.round(prob * 100)}%`;
            wrapper.appendChild(badge);
        }
        container.appendChild(wrapper);
    }

    function formatLabel(str) {
        if (!str) return '';
        return str.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }

    // High-Precision Video Frame Synchronization
    function updateSkeletonCanvasDimensions() {
        const dpr = window.devicePixelRatio || 1;
        const rect = skeletonCanvas.getBoundingClientRect();
        const targetWidth = Math.round(rect.width * dpr);
        const targetHeight = Math.round(rect.height * dpr);

        if (skeletonCanvas.width !== targetWidth || skeletonCanvas.height !== targetHeight) {
            skeletonCanvas.width = targetWidth;
            skeletonCanvas.height = targetHeight;
        }
    }

    function syncSkeletonToVideo(mediaTime) {
        if (!sourceVideoView || !sourceVideoView.duration || state.poseHistory.length === 0) return;
        
        const currentTime = (typeof mediaTime === 'number') ? mediaTime : sourceVideoView.currentTime;
        const progress = Math.max(0, Math.min(1, currentTime / sourceVideoView.duration));
        const frameIdx = Math.min(
            Math.round(progress * (state.poseHistory.length - 1)),
            state.poseHistory.length - 1
        );

        if (frameIdx >= 0 && frameIdx !== state.currentFrame) {
            state.currentFrame = frameIdx;
            timelineScrubber.value = frameIdx;
            updateFrameCounterText();
            drawSkeletonFrame(frameIdx);
        }
    }

    function videoFrameCallback(now, metadata) {
        if (sourceVideoView && !sourceVideoView.paused) {
            syncSkeletonToVideo(metadata ? metadata.mediaTime : undefined);
            if ('requestVideoFrameCallback' in sourceVideoView) {
                sourceVideoView.requestVideoFrameCallback(videoFrameCallback);
            }
        }
    }

    function fallbackRenderLoop() {
        if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype) && sourceVideoView && !sourceVideoView.paused) {
            syncSkeletonToVideo();
        }
        requestAnimationFrame(fallbackRenderLoop);
    }
    requestAnimationFrame(fallbackRenderLoop);

    if (sourceVideoView) {
        sourceVideoView.addEventListener('play', () => {
            if ('requestVideoFrameCallback' in sourceVideoView) {
                sourceVideoView.requestVideoFrameCallback(videoFrameCallback);
            }
        });
        sourceVideoView.addEventListener('seeked', () => {
            syncSkeletonToVideo();
        });
        sourceVideoView.addEventListener('timeupdate', () => {
            if (sourceVideoView.paused) {
                syncSkeletonToVideo();
            }
        });
    }

    timelineScrubber.addEventListener('input', () => {
        if (!sourceVideoView.duration || state.poseHistory.length === 0) return;
        state.currentFrame = parseInt(timelineScrubber.value);
        const targetTime = (state.currentFrame / (state.poseHistory.length - 1)) * sourceVideoView.duration;
        sourceVideoView.currentTime = targetTime;
        updateFrameCounterText();
        drawSkeletonFrame(state.currentFrame);
    });

    playPauseBtn.addEventListener('click', () => {
        if (sourceVideoView.paused) {
            startSkeletonPlay();
        } else {
            pauseSkeleton();
        }
    });

    // Keyboard Shortcuts for Frame Stepping
    document.addEventListener('keydown', (e) => {
        if (!state.poseHistory || state.poseHistory.length === 0) return;
        if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;

        if (e.code === 'Space') {
            e.preventDefault();
            if (sourceVideoView.paused) {
                startSkeletonPlay();
            } else {
                pauseSkeleton();
            }
        } else if (e.code === 'ArrowLeft' || e.key === ',') {
            e.preventDefault();
            pauseSkeleton();
            const prevFrame = Math.max(0, state.currentFrame - 1);
            seekToFrame(prevFrame);
        } else if (e.code === 'ArrowRight' || e.key === '.') {
            e.preventDefault();
            pauseSkeleton();
            const nextFrame = Math.min(state.poseHistory.length - 1, state.currentFrame + 1);
            seekToFrame(nextFrame);
        }
    });

    function seekToFrame(frameIdx) {
        if (!sourceVideoView.duration || state.poseHistory.length === 0) return;
        state.currentFrame = frameIdx;
        timelineScrubber.value = frameIdx;
        const targetTime = (frameIdx / (state.poseHistory.length - 1)) * sourceVideoView.duration;
        sourceVideoView.currentTime = targetTime;
        updateFrameCounterText();
        drawSkeletonFrame(frameIdx);
    }

    function startSkeletonPlay() {
        sourceVideoView.play();
        playPauseBtn.innerText = "⏸";
        if ('requestVideoFrameCallback' in sourceVideoView) {
            sourceVideoView.requestVideoFrameCallback(videoFrameCallback);
        }
    }

    function pauseSkeleton() {
        sourceVideoView.pause();
        playPauseBtn.innerText = "▶";
    }

    function updateFrameCounterText() {
        frameCounter.innerText = `${state.currentFrame + 1} / ${state.poseHistory.length}`;
    }

    function drawSkeletonFrame(frameIdx) {
        if (!state.poseHistory || state.poseHistory.length === 0) return;
        const landmarks = state.poseHistory[frameIdx];
        if (!landmarks) return;

        updateSkeletonCanvasDimensions();
        const dpr = window.devicePixelRatio || 1;
        const rect = skeletonCanvas.getBoundingClientRect();
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        ctx.clearRect(0, 0, rect.width, rect.height);

        let renderWidth = rect.width;
        let renderHeight = rect.height;
        let offsetX = 0;
        let offsetY = 0;

        if (sourceVideoView && sourceVideoView.videoWidth > 0 && sourceVideoView.videoHeight > 0) {
            const containerAspect = rect.width / rect.height;
            const videoAspect = sourceVideoView.videoWidth / sourceVideoView.videoHeight;

            if (videoAspect > containerAspect) {
                renderWidth = rect.width;
                renderHeight = rect.width / videoAspect;
                offsetY = (rect.height - renderHeight) / 2;
            } else {
                renderHeight = rect.height;
                renderWidth = rect.height * videoAspect;
                offsetX = (rect.width - renderWidth) / 2;
            }
        }

        const getCanvasCoord = (pt) => {
            if (!pt) return null;
            return {
                x: offsetX + pt.x * renderWidth,
                y: offsetY + pt.y * renderHeight
            };
        };

        const connections = [
            ['left_shoulder', 'right_shoulder'],
            ['left_shoulder', 'left_elbow'],
            ['left_elbow', 'left_wrist'],
            ['right_shoulder', 'right_elbow'],
            ['right_elbow', 'right_wrist'],
            ['left_shoulder', 'left_hip'],
            ['right_shoulder', 'right_hip'],
            ['left_hip', 'right_hip'],
            ['left_hip', 'left_knee'],
            ['left_knee', 'left_ankle'],
            ['right_hip', 'right_knee'],
            ['right_knee', 'right_ankle']
        ];

        // Draw bone lines (bright green)
        ctx.strokeStyle = '#00ff7f';
        ctx.lineWidth = 3;
        ctx.shadowBlur = 8;
        ctx.shadowColor = 'rgba(0, 255, 127, 0.6)';
        ctx.lineCap = 'round';

        connections.forEach(([ptA, ptB]) => {
            const posA = getCanvasCoord(landmarks[ptA]);
            const posB = getCanvasCoord(landmarks[ptB]);
            if (posA && posB) {
                ctx.beginPath();
                ctx.moveTo(posA.x, posA.y);
                ctx.lineTo(posB.x, posB.y);
                ctx.stroke();
            }
        });

        // Draw joint dots
        ctx.shadowBlur = 6;
        ctx.shadowColor = 'rgba(0, 100, 255, 0.8)';
        ctx.fillStyle = '#0064ff';

        for (const jointName in landmarks) {
            const pos = getCanvasCoord(landmarks[jointName]);
            if (!pos) continue;

            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
    }

    window.addEventListener('resize', () => {
        if (state.poseHistory.length > 0) {
            drawSkeletonFrame(state.currentFrame);
        }
    });

    // Render Joint Angles Chart using Chart.js
    function renderAnglesChart(metrics, isImage) {
        anglesEmptyState.classList.add('hidden');
        chartContainer.classList.remove('hidden');

        let labels = [];
        let values = [];
        
        if (!metrics) metrics = {};

        if (isImage) {
            labels = [
                'Right Elbow', 'Left Elbow', 
                'Right Shoulder', 'Left Shoulder', 
                'Right Knee', 'Left Knee', 
                'Trunk Rotation', 'Trunk Lean'
            ];
            values = [
                metrics.right_elbow_angle !== undefined ? metrics.right_elbow_angle : (metrics.right_elbow || 0),
                metrics.left_elbow_angle !== undefined ? metrics.left_elbow_angle : (metrics.left_elbow || 0),
                metrics.right_shoulder_angle !== undefined ? metrics.right_shoulder_angle : (metrics.right_shoulder || 0),
                metrics.left_shoulder_angle !== undefined ? metrics.left_shoulder_angle : (metrics.left_shoulder || 0),
                metrics.right_knee_angle !== undefined ? metrics.right_knee_angle : (metrics.right_knee || 0),
                metrics.left_knee_angle !== undefined ? metrics.left_knee_angle : (metrics.left_knee || 0),
                metrics.trunk_rotation_diff !== undefined ? metrics.trunk_rotation_diff : (metrics.trunk_rotation || 0),
                metrics.trunk_lean !== undefined ? (Math.abs(metrics.trunk_lean) < 3.2 ? Math.round(metrics.trunk_lean * 57.3 * 10) / 10 : metrics.trunk_lean) : 0
            ];
        } else {
            labels = [
                'Avg Right Elbow', 'Avg Left Elbow', 
                'Avg Right Shoulder', 'Avg Left Shoulder', 
                'Avg Right Knee', 'Avg Left Knee', 
                'Avg Trunk Rotation', 'Avg Trunk Lean'
            ];
            values = [
                metrics.avg_right_elbow_angle !== undefined ? metrics.avg_right_elbow_angle : (metrics.right_elbow || 0),
                metrics.avg_left_elbow_angle !== undefined ? metrics.avg_left_elbow_angle : (metrics.left_elbow || 0),
                metrics.avg_right_shoulder_angle !== undefined ? metrics.avg_right_shoulder_angle : (metrics.right_shoulder || 0),
                metrics.avg_left_shoulder_angle !== undefined ? metrics.avg_left_shoulder_angle : (metrics.left_shoulder || 0),
                metrics.avg_right_knee_angle !== undefined ? metrics.avg_right_knee_angle : (metrics.right_knee || 0),
                metrics.avg_left_knee_angle !== undefined ? metrics.avg_left_knee_angle : (metrics.left_knee || 0),
                metrics.avg_trunk_rotation_diff !== undefined ? metrics.avg_trunk_rotation_diff : (metrics.trunk_rotation || 0),
                metrics.avg_trunk_lean !== undefined ? (Math.abs(metrics.avg_trunk_lean) < 3.2 ? Math.round(metrics.avg_trunk_lean * 57.3 * 10) / 10 : metrics.avg_trunk_lean) : (metrics.trunk_lean || 0)
            ];
        }

        if (state.metricsChart) {
            state.metricsChart.destroy();
        }

        const ctxChart = document.getElementById('metrics-chart').getContext('2d');
        state.metricsChart = new Chart(ctxChart, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Degrees (°)',
                    data: values,
                    backgroundColor: 'rgba(0, 255, 127, 0.25)',
                    borderColor: 'rgba(0, 255, 127, 0.85)',
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.raw.toFixed(1)}°`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af' },
                        suggestedMax: 180
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#9ca3af' }
                    }
                }
            }
        });
    }
});
