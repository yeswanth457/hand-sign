/**
 * RT-STAMP-SLR Client App Logic — Browser-Side MediaPipe Architecture
 * 
 * Architecture:
 *   Webcam → Browser MediaPipe (HandLandmarker + PoseLandmarker) → Canvas skeleton (instant)
 *   → 6D token → FastAPI /api/process_token → CNN-GRU → Translation
 *
 * Key design decisions:
 *   - MediaPipe runs IN the browser for zero-latency landmark visualization
 *   - Only the 6D token is sent to the backend (not JPEG frames)
 *   - Latest-frame-wins: stale landmarks are never drawn over newer frames
 *   - Correct 21-point MediaPipe hand skeleton with 20 bone connections
 */

import { FilesetResolver, HandLandmarker, PoseLandmarker } from
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ─── Explicit Pipeline States (Phase 11) ──────────────────────────
// States: CAMERA_OFF, CAMERA_STARTING, CAMERA_ERROR, NO_HAND, COLLECTING, READY, PREDICTING, ACCEPTED
let currentSystemState = 'CAMERA_OFF';
let isWebcamRunning = false;
let mediaStream = null;
let currentLanguage = 'english';
let motionHistory = new Array(50).fill(0);
let thresholdHistory = new Array(50).fill(0.015);
let frameId = 0;
let lastTranslationText = "";
let currentPrediction = { word: '--', confidence: 0.0 };
let currentEarlyDecision = { state: 'CAMERA_OFF' };

// Single Gesture Test Mode (Phase 10)
let singleTestTargetSign = 'hello';
let singleTestState = 'READY';
let lastSingleAcceptedSign = '--';

// MediaPipe browser-side instances
let handLandmarker = null;
let poseLandmarker = null;
let mediaPipeReady = false;

// Timing / diagnostics (Phase 8 Performance Tracking)
let isProcessingToken = false;
let processedFramesCount = 0;
let droppedFramesCount = 0;
let lastFrameTimestamp = performance.now();
let webcamFps = 0;
let mpHandInferenceMs = 0;
let mpPoseInferenceMs = 0;
let tokenCalculationMs = 0;
let renderDurationMs = 0;
let roundTripLatencyMs = 0;
let latestRenderedFrameId = 0;
let _lastCameraDebugLog = 0;
let _lastHandDebugLog = 0;
let _lastDetectError = null;
let _detectCallCount = 0;
let _detectErrorCount = 0;
let _lastNoWebLog = 0;

let latestDebugMetrics = {
    handsCount: 0,
    poseStatus: 'Inactive',
    landmarksCount: 0,
    queueStatus: 'LATEST / 0 BACKLOG',
    bufferStatus: '0/25',
    mpMode: 'Browser-Side'
};

// ─── MediaPipe Hand Skeleton Connections (Phase 7 Topology) ───────
// Correct 21-point MediaPipe hand topology with 20 bone segments + palm base
const HAND_CONNECTIONS = [
    // Thumb: Wrist(0) → CMC(1) → MCP(2) → IP(3) → TIP(4)
    [0, 1], [1, 2], [2, 3], [3, 4],
    // Index: Wrist(0) → MCP(5) → PIP(6) → DIP(7) → TIP(8)
    [0, 5], [5, 6], [6, 7], [7, 8],
    // Middle: Wrist(0) → MCP(9) → PIP(10) → DIP(11) → TIP(12)
    [0, 9], [9, 10], [10, 11], [11, 12],
    // Ring: Wrist(0) → MCP(13) → PIP(14) → DIP(15) → TIP(16)
    [0, 13], [13, 14], [14, 15], [15, 16],
    // Pinky: Wrist(0) → MCP(17) → PIP(18) → DIP(19) → TIP(20)
    [0, 17], [17, 18], [18, 19], [19, 20],
    // Palm Base & Knuckles
    [5, 9], [9, 13], [13, 17]
];

// Fingertip landmark indices for distinctive highlights (#4, #8, #12, #16, #20)
const FINGERTIP_INDICES = [4, 8, 12, 16, 20];

// ─── DOM Elements ────────────────────────────────────────────────
const videoEl = document.getElementById('webcamVideo');
const canvasEl = document.getElementById('landmarkCanvas');
const ctx = canvasEl.getContext('2d');
const motionGraphCanvas = document.getElementById('motionGraphCanvas');
const graphCtx = motionGraphCanvas.getContext('2d');

const btnToggleWebcam = document.getElementById('btnToggleWebcam');
const btnSimulateDemo = document.getElementById('btnSimulateDemo');
const btnSimulateSelected = document.getElementById('btnSimulateSelected');
const btnTrainModel = document.getElementById('btnTrainModel');
const btnTTS = document.getElementById('btnTTS');
const btnClear = document.getElementById('btnClear');

const selectVocab = document.getElementById('selectVocab');
const valMotionEnergy = document.getElementById('valMotionEnergy');
const decisionStateBadge = document.getElementById('decisionStateBadge');

// Debug HUD Elements (Phase 3)
const dbgCameraStream = document.getElementById('dbgCameraStream');
const dbgReadyState = document.getElementById('dbgReadyState');
const dbgDimensions = document.getElementById('dbgDimensions');
const dbgCurrentTime = document.getElementById('dbgCurrentTime');
const dbgLoopStatus = document.getElementById('dbgLoopStatus');
const dbgFrameId = document.getElementById('dbgFrameId');
const dbgDetectorStatus = document.getElementById('dbgDetectorStatus');
const dbgHandDetected = document.getElementById('dbgHandDetected');
const dbgLandmarkCount = document.getElementById('dbgLandmarkCount');
const dbgFpsPill = document.getElementById('dbgFpsPill');

const valMpHandTime = document.getElementById('valMpHandTime');
const valMpPoseTime = document.getElementById('valMpPoseTime');
const valRenderTime = document.getElementById('valRenderTime');
const valTokenCalcTime = document.getElementById('valTokenCalcTime');
const valBackendRtt = document.getElementById('valBackendRtt');

// Single Gesture Test Mode Elements (Phase 10)
const sgTargetButtons = document.querySelectorAll('.sg-target-btn');
const btnClearSgTest = document.getElementById('btnClearSgTest');
const sgValSign = document.getElementById('sgValSign');
const sgValConf = document.getElementById('sgValConf');
const sgValState = document.getElementById('sgValState');

// 6D Token Boxes
const tkHx = document.getElementById('tkHx');
const tkHy = document.getElementById('tkHy');
const tkMx = document.getElementById('tkMx');
const tkMy = document.getElementById('tkMy');
const tkRx = document.getElementById('tkRx');
const tkRy = document.getElementById('tkRy');

// Translation Output
const currentLangLabel = document.getElementById('currentLangLabel');
const translatedText = document.getElementById('translatedText');
const wordBufferContainer = document.getElementById('wordBufferContainer');
const valDetectedWord = document.getElementById('valDetectedWord');
const valConfidencePct = document.getElementById('valConfidencePct');
const confidenceBarFill = document.getElementById('confidenceBarFill');
const valThreshold = document.getElementById('valThreshold');

// ─── Camera State (separate from pipeline/gesture state) ────────
let currentCameraState = 'CAMERA_OFF';
let currentPipelineState = 'PIPELINE_STARTING';

function setCameraState(newState, reason) {
    currentCameraState = newState;
    const el = document.getElementById('dbgCameraState');
    if (el) {
        el.textContent = newState;
        el.className = 'dbg-val ' + (
            newState === 'CAMERA_LIVE' ? 'val-active' :
            newState === 'CAMERA_STARTING' ? 'val-warning' :
            newState === 'CAMERA_ERROR' ? 'val-error' : 'val-inactive'
        );
    }
    if (reason) console.log(`[Camera State] ${newState}: ${reason}`);
}

function setPipelineState(newState, reason) {
    currentPipelineState = newState;
    const el = document.getElementById('dbgPipelineState');
    if (el) {
        el.textContent = newState;
        el.className = 'dbg-val ' + (
            newState === 'PIPELINE_READY' ? 'val-active' :
            newState === 'PIPELINE_ERROR' ? 'val-error' : 'val-warning'
        );
    }
    if (reason) console.log(`[Pipeline State] ${newState}: ${reason}`);
}

// ─── Gesture/System State Machine Transition Helper (Phase 11) ──
function updateSystemState(newState, reason) {
    currentSystemState = newState;
    if (decisionStateBadge) {
        decisionStateBadge.className = 'decision-badge';
        if (newState === 'CAMERA_OFF') {
            decisionStateBadge.textContent = 'CAMERA_OFF';
            decisionStateBadge.classList.add('state-camera-off');
        } else if (newState === 'CAMERA_STARTING') {
            decisionStateBadge.textContent = 'CAMERA_STARTING';
            decisionStateBadge.classList.add('state-camera-starting');
        } else if (newState === 'CAMERA_ERROR') {
            decisionStateBadge.textContent = 'CAMERA_ERROR';
            decisionStateBadge.classList.add('state-camera-error');
        } else if (newState === 'NO_HAND') {
            decisionStateBadge.textContent = 'WAITING FOR HAND GESTURE';
            decisionStateBadge.classList.add('state-no-hand');
        } else if (newState === 'COLLECTING') {
            decisionStateBadge.textContent = 'COLLECTING GESTURE...';
            decisionStateBadge.classList.add('state-collecting');
        } else if (newState === 'READY') {
            decisionStateBadge.textContent = 'READY';
            decisionStateBadge.classList.add('state-ready');
        } else if (newState === 'PREDICTING') {
            decisionStateBadge.textContent = 'PREDICTING';
            decisionStateBadge.classList.add('state-predicting');
        } else if (newState === 'ACCEPTED') {
            decisionStateBadge.textContent = 'ACCEPTED';
            decisionStateBadge.classList.add('state-accepted');
        } else if (newState === 'COOLDOWN') {
            decisionStateBadge.textContent = 'COOLDOWN';
            decisionStateBadge.classList.add('state-cooldown');
        } else {
            decisionStateBadge.textContent = newState;
        }
    }

    if (sgValState) {
        sgValState.textContent = newState;
        sgValState.className = 'sg-state-tag' + (newState === 'ACCEPTED' ? ' accepted' : '');
    }

    if (reason) {
        console.log(`[Gesture State] ${newState}: ${reason}`);
    }
}

// ─── Init ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setCameraState('CAMERA_OFF', 'Application initialized, camera idle.');
    setPipelineState('PIPELINE_STARTING', 'Initializing MediaPipe...');
    updateSystemState('CAMERA_OFF', 'Application initialized, camera idle.');

    // Initial clean UI state
    if (valDetectedWord) valDetectedWord.textContent = '--';
    if (valConfidencePct) valConfidencePct.textContent = '0%';
    if (confidenceBarFill) confidenceBarFill.style.width = '0%';
    if (valMotionEnergy) valMotionEnergy.textContent = '0.0000';
    if (translatedText) translatedText.innerHTML = '<em>WAITING FOR WEBCAM</em>';
    updateWordBuffer([]);
    if (sgValSign) sgValSign.textContent = '--';
    if (sgValConf) sgValConf.textContent = '0.0%';
    if (sgValState) {
        sgValState.textContent = 'CAMERA_OFF';
        sgValState.className = 'sg-state-tag';
    }

    loadVocabulary();
    initEventListeners();
    drawMotionGraph();
    initMediaPipe();

    // Ensure backend sentence buffer is clear on page load
    fetch('/api/clear_sentence', { method: 'POST' }).catch(() => {});
});

function initEventListeners() {
    if (btnToggleWebcam) btnToggleWebcam.addEventListener('click', toggleWebcam);
    if (btnSimulateDemo) btnSimulateDemo.addEventListener('click', () => simulateSign('water'));
    if (btnSimulateSelected) btnSimulateSelected.addEventListener('click', () => {
        const word = selectVocab ? selectVocab.value : null;
        if (word) simulateSign(word);
    });
    if (btnTrainModel) btnTrainModel.addEventListener('click', trainModel);
    if (btnTTS) btnTTS.addEventListener('click', speakTranslation);
    if (btnClear) btnClear.addEventListener('click', clearSentence);

    // Single Gesture Test buttons
    sgTargetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.target;
            if (!target) return;
            singleTestTargetSign = target;
            sgTargetButtons.forEach(b => {
                if (b.dataset.target) b.classList.toggle('active', b.dataset.target === target);
            });
            console.log(`[Single Gesture Mode] Target set to: ${target.toUpperCase()}`);
        });
    });

    if (btnClearSgTest) {
        btnClearSgTest.addEventListener('click', resetSingleGestureTest);
    }
}

function resetSingleGestureTest() {
    currentPrediction = { word: '--', confidence: 0.0 };
    if (sgValSign) sgValSign.textContent = '--';
    if (sgValConf) sgValConf.textContent = '0.0%';
    if (sgValState) {
        sgValState.textContent = isWebcamRunning ? 'READY' : 'CAMERA_OFF';
        sgValState.className = 'sg-state-tag';
    }
    clearSentence();
    console.log('[Single Gesture Mode] Test display reset.');
}

// ─── MediaPipe Browser-Side Initialization (non-blocking) ───────
async function initMediaPipe() {
    setPipelineState('PIPELINE_STARTING', 'Loading MediaPipe WASM + models...');
    if (dbgDetectorStatus) dbgDetectorStatus.textContent = 'STARTING';

    // Try local models first (/models/ served by FastAPI), then CDN fallback
    const localHandModel = '/models/hand_landmarker.task';
    const cdnHandModel = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
    const localPoseModel = '/models/pose_landmarker.task';
    const cdnPoseModel = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );

        // Try GPU with local model, fall through on any failure
        let handModel = localHandModel;
        let poseModel = localPoseModel;
        let delegate = 'GPU';

        try {
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: { modelAssetPath: handModel, delegate },
                runningMode: 'VIDEO', numHands: 2,
                minHandDetectionConfidence: 0.40, minHandPresenceConfidence: 0.40, minTrackingConfidence: 0.30
            });
        } catch (eLocalHand) {
            console.warn('[MediaPipe] Local hand model failed, trying CDN:', eLocalHand.message);
            handModel = cdnHandModel;
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: { modelAssetPath: handModel, delegate },
                runningMode: 'VIDEO', numHands: 2,
                minHandDetectionConfidence: 0.40, minHandPresenceConfidence: 0.40, minTrackingConfidence: 0.30
            });
        }

        try {
            poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
                baseOptions: { modelAssetPath: poseModel, delegate },
                runningMode: 'VIDEO', numPoses: 1,
                minPoseDetectionConfidence: 0.30, minPosePresenceConfidence: 0.30, minTrackingConfidence: 0.30
            });
        } catch (eLocalPose) {
            console.warn('[MediaPipe] Local pose model failed, trying CDN:', eLocalPose.message);
            poseModel = cdnPoseModel;
            poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
                baseOptions: { modelAssetPath: poseModel, delegate },
                runningMode: 'VIDEO', numPoses: 1,
                minPoseDetectionConfidence: 0.30, minPosePresenceConfidence: 0.30, minTrackingConfidence: 0.30
            });
        }

        mediaPipeReady = true;
        setPipelineState('PIPELINE_READY', `Browser-side Hand+Pose initialized (${delegate}, hand=${handModel.includes('/models/') ? 'local' : 'CDN'})`);
        if (dbgDetectorStatus) dbgDetectorStatus.textContent = `READY (${delegate})`;
    } catch (eGpu) {
        console.warn('[MediaPipe] GPU failed, trying CPU fallback:', eGpu.message);
        try {
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
            );
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: { modelAssetPath: cdnHandModel, delegate: 'CPU' },
                runningMode: 'VIDEO', numHands: 2,
                minHandDetectionConfidence: 0.40, minHandPresenceConfidence: 0.40, minTrackingConfidence: 0.30
            });
            poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
                baseOptions: { modelAssetPath: cdnPoseModel, delegate: 'CPU' },
                runningMode: 'VIDEO', numPoses: 1,
                minPoseDetectionConfidence: 0.30, minPosePresenceConfidence: 0.30, minTrackingConfidence: 0.30
            });
            mediaPipeReady = true;
            setPipelineState('PIPELINE_READY', 'Browser-side Hand+Pose initialized (CPU, CDN)');
            if (dbgDetectorStatus) dbgDetectorStatus.textContent = 'READY (CPU)';
        } catch (eCpu) {
            console.error('[MediaPipe] All initialization attempts failed:', eCpu);
            mediaPipeReady = false;
            setPipelineState('PIPELINE_ERROR', `MediaPipe init failed: ${eCpu.message}`);
            if (dbgDetectorStatus) {
                dbgDetectorStatus.textContent = `ERROR: ${eCpu.name || 'InitFailed'}`;
                dbgDetectorStatus.className = 'dbg-val val-error';
            }
        }
    }
}

// ─── Vocabulary Loader ───────────────────────────────────────────
async function loadVocabulary() {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);
    try {
        const res = await fetch('/api/vocabulary', { signal: controller.signal });
        clearTimeout(timeoutId);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (selectVocab && data && Array.isArray(data.vocabulary)) {
            selectVocab.innerHTML = '';
            data.vocabulary.forEach(item => {
                const opt = document.createElement('option');
                opt.value = item.english;
                opt.textContent = `${item.id + 1}. ${item.display_english} → ${item.tamil}`;
                selectVocab.appendChild(opt);
            });
        }
        console.log(`[VOCABULARY] Successfully loaded ${data?.vocabulary?.length || 0} vocabulary items.`);
    } catch (e) {
        clearTimeout(timeoutId);
        console.error('Failed to load vocabulary:', e);
    }
}

// ─── Robust Webcam Startup & Lifecycle (Phase 2) ─────────────────
async function toggleWebcam() {
    if (isWebcamRunning) {
        stopWebcam();
    } else {
        await startWebcam();
    }
}

async function startWebcam() {
    if (isWebcamRunning) return;

    // Reset live state before processing new frames
    currentPrediction = { word: '--', confidence: 0.0 };
    currentEarlyDecision = { state: 'CAMERA_STARTING' };
    lastTranslationText = "";
    motionHistory = new Array(50).fill(0);
    thresholdHistory = new Array(50).fill(0.015);
    drawMotionGraph();

    if (valDetectedWord) valDetectedWord.textContent = '--';
    if (valConfidencePct) valConfidencePct.textContent = '0%';
    if (confidenceBarFill) confidenceBarFill.style.width = '0%';
    if (valMotionEnergy) valMotionEnergy.textContent = '0.0000';

    if (sgValSign) sgValSign.textContent = '--';
    if (sgValConf) sgValConf.textContent = '0.0%';
    if (sgValState) {
        sgValState.textContent = 'STARTING';
        sgValState.className = 'sg-state-tag';
    }

    if (translatedText) {
        translatedText.innerHTML = '<em>WAITING FOR HAND GESTURE</em>';
    }
    updateWordBuffer([]);
    updateTokenDisplay(null, false);

    // Reset backend server session buffer
    try {
        await fetch('/api/clear_sentence', { method: 'POST' });
    } catch (_) {}

    setCameraState('CAMERA_STARTING', 'Requesting camera device access...');
    updateSystemState('CAMERA_STARTING', 'Requesting camera device access...');
    btnToggleWebcam.disabled = true;
    btnToggleWebcam.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Starting...';

    try {
        // 2. Verify browser API support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('navigator.mediaDevices.getUserMedia is not supported by this browser environment');
        }

        // 3. Request camera MediaStream
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });

        // 4. Verify stream validity
        if (!mediaStream || !mediaStream.active) {
            throw new Error('Retrieved MediaStream is inactive or empty');
        }

        // Handle stream track interruptions
        mediaStream.getVideoTracks().forEach(track => {
            track.addEventListener('ended', () => {
                console.warn('[Webcam] Camera track was terminated.');
                stopWebcam();
            });
        });

        // 5. Assign stream to HTML video element
        videoEl.srcObject = mediaStream;

        // 6. Await video metadata to guarantee videoWidth & videoHeight > 0
        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Timed out waiting for video metadata (8s)'));
            }, 8000);

            if (videoEl.readyState >= HTMLMediaElement.HAVE_METADATA && videoEl.videoWidth > 0) {
                clearTimeout(timeout);
                resolve();
            } else {
                videoEl.onloadedmetadata = () => {
                    clearTimeout(timeout);
                    resolve();
                };
            }
        });

        // 7. Invoke video.play()
        await videoEl.play();

        // 8. Verify readyState >= HAVE_CURRENT_DATA (readyState >= 2)
        if (videoEl.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            await new Promise((resolve, reject) => {
                const interval = setInterval(() => {
                    if (videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
                        clearInterval(interval);
                        resolve();
                    }
                }, 50);
                setTimeout(() => {
                    clearInterval(interval);
                    if (videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) resolve();
                    else reject(new Error('Video readyState failed to reach HAVE_CURRENT_DATA'));
                }, 5000);
            });
        }

        // 9. Verify positive video dimensions
        if (videoEl.videoWidth <= 0 || videoEl.videoHeight <= 0) {
            throw new Error(`Invalid video stream dimensions: ${videoEl.videoWidth}x${videoEl.videoHeight}`);
        }

        // 10. Synchronize canvas resolution 1:1 with video source dimensions
        canvasEl.width = videoEl.videoWidth;
        canvasEl.height = videoEl.videoHeight;

        isWebcamRunning = true;
        btnToggleWebcam.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Webcam';
        btnToggleWebcam.style.background = '#ef4444';
        btnToggleWebcam.disabled = false;

        setCameraState('CAMERA_LIVE', `Webcam active (${videoEl.videoWidth}x${videoEl.videoHeight})`);
        updateSystemState('NO_HAND', `Webcam active. Waiting for hand gesture.`);
        if (translatedText) {
            translatedText.innerHTML = '<em>WAITING FOR HAND GESTURE</em>';
        }
        console.log(`[Webcam] Initialized successfully. Source: ${videoEl.videoWidth}x${videoEl.videoHeight}, readyState: ${videoEl.readyState}`);

        // 11. Launch continuous processing loop
        requestAnimationFrame(processWebcamLoop);

    } catch (err) {
        console.error('[Webcam] Startup failed:', err);
        stopWebcam();
        btnToggleWebcam.disabled = false;

        let friendlyMsg = err.message || 'Unknown camera error';
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            friendlyMsg = 'Camera access was denied. Please grant camera permission in your browser address bar.';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            friendlyMsg = 'No camera hardware found on this computer.';
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
            friendlyMsg = 'Camera is currently locked by another application.';
        }

        setCameraState('CAMERA_ERROR', friendlyMsg);
        updateSystemState('CAMERA_ERROR', friendlyMsg);
        alert(`Camera Error: ${friendlyMsg}`);
    }
}

function stopWebcam() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => {
            try { track.stop(); } catch (_) { }
        });
        mediaStream = null;
    }
    videoEl.srcObject = null;
    isWebcamRunning = false;
    btnToggleWebcam.innerHTML = '<i class="fa-solid fa-video"></i> Start Webcam';
    btnToggleWebcam.style.background = '#3b82f6';
    btnToggleWebcam.disabled = false;

    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    setCameraState('CAMERA_OFF', 'Webcam stopped by user.');
    updateSystemState('CAMERA_OFF', 'Webcam stopped by user.');

    // Clear live predictions & metrics immediately
    currentPrediction = { word: '--', confidence: 0.0 };
    currentEarlyDecision = { state: 'CAMERA_OFF' };
    lastTranslationText = "";

    if (valDetectedWord) valDetectedWord.textContent = '--';
    if (valConfidencePct) valConfidencePct.textContent = '0%';
    if (confidenceBarFill) confidenceBarFill.style.width = '0%';
    if (valMotionEnergy) valMotionEnergy.textContent = '0.0000';

    if (sgValSign) sgValSign.textContent = '--';
    if (sgValConf) sgValConf.textContent = '0.0%';
    if (sgValState) {
        sgValState.textContent = 'CAMERA_OFF';
        sgValState.className = 'sg-state-tag';
    }

    if (translatedText) {
        translatedText.innerHTML = '<em>WAITING FOR WEBCAM</em>';
    }
    updateWordBuffer([]);

    updateTokenDisplay(null, false);
    updateDebugHUDLiveMetrics(false, 0, 0, null);

    // Call backend to clear server-side session buffers
    fetch('/api/clear_sentence', { method: 'POST' }).catch(() => {});
}

// ─── Main Frame Processing Loop ─────────────────────────────────
// Architecture:
//   1. Browser MediaPipe runs on current video frame → landmarks (instant)
//   2. Canvas draws skeleton on same frame (zero visual latency)
//   3. 6D token computed in browser
//   4. Token sent to backend for CNN-GRU inference (async, non-blocking for rendering)
//
// Latest-frame-wins: landmarks always match the visible video frame.
// ─── Main Frame Processing Loop (Phases 3, 4, 5, 8) ──────────────
let lastVideoTimestampMs = -1;
let cachedPoseResults = null;
let poseRunInterval = 3;  // Run Pose every Nth frame

async function processWebcamLoop() {
    if (!isWebcamRunning) {
        updateDebugHUDLiveMetrics(false, 0, 0);
        return;
    }

    // PHASE 18: Outer defensive try/catch — an uncaught exception here would kill the rAF loop
    try {
        const now = performance.now();
        const dtFrame = now - lastFrameTimestamp;
        lastFrameTimestamp = now;
        webcamFps = Math.round(1000 / Math.max(1, dtFrame));

        frameId++;

        if (mediaPipeReady && videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
            // Monotonically increasing timestamp required for MediaPipe VIDEO mode
            // Use Math.floor + 1 to guarantee strict monotonicity even at high refresh rates
            const videoTimestampMs = Math.max(lastVideoTimestampMs + 1, Math.round(now));
            lastVideoTimestampMs = videoTimestampMs;

            // Ensure canvas matches video source dimensions
            if (videoEl.videoWidth && (canvasEl.width !== videoEl.videoWidth || canvasEl.height !== videoEl.videoHeight)) {
                canvasEl.width = videoEl.videoWidth;
                canvasEl.height = videoEl.videoHeight;
            }

            let handResults = null;
            let poseResults = null;
            let handDetectError = null;
            let poseDetectError = null;

            // Phase 8: Measure MP Hand inference time separately
            const tHandStart = performance.now();
            _detectCallCount++;
            try {
                handResults = handLandmarker.detectForVideo(videoEl, videoTimestampMs);
            } catch (e) {
                handDetectError = e;
                _lastDetectError = e;
                _detectErrorCount++;
            }
            mpHandInferenceMs = performance.now() - tHandStart;

            // Pose inference using PoseLandmarker (every poseRunInterval frames to maintain high FPS)
            const tPoseStart = performance.now();
            if (poseLandmarker && (frameId % poseRunInterval === 0 || !cachedPoseResults)) {
                try {
                    cachedPoseResults = poseLandmarker.detectForVideo(videoEl, videoTimestampMs);
                } catch (e) {
                    cachedPoseResults = null;
                }
            }
            poseResults = cachedPoseResults;
            mpPoseInferenceMs = performance.now() - tPoseStart;

            // Phase 6: Build landmark data with STRICT single-hand filtering (21 landmarks)
            const tTokenStart = performance.now();
            const landmarkData = buildLandmarkData(handResults, poseResults);
            tokenCalculationMs = performance.now() - tTokenStart;

            const hasHand = landmarkData.hands && landmarkData.hands.length > 0;
            const handCount = hasHand ? landmarkData.hands.length : 0;
            const lmCount = handCount * 21;

            // Phase 7: Draw clean skeleton on transparent canvas (zero latency, zero obscuration)
            latestRenderedFrameId = frameId;
            const renderStart = performance.now();
            drawRealLandmarks(landmarkData);
            renderDurationMs = performance.now() - renderStart;

            // Phase 3: Update Live Debug HUD in browser DOM with dual-hand metadata
            updateDebugHUDLiveMetrics(hasHand, handCount, lmCount, landmarkData);

            // Phase 11: Maintain correct pipeline state
            if (!hasHand) {
                if (currentSystemState !== 'CAMERA_OFF' && currentSystemState !== 'CAMERA_STARTING') {
                    updateSystemState('NO_HAND');
                }
            } else {
                if (currentSystemState === 'NO_HAND' || currentSystemState === 'CAMERA_OFF') {
                    updateSystemState('COLLECTING', 'Hand detected, collecting frames');
                }
            }

            // ═══════════════════════════════════════════════════════════
            // PHASE 4 & 17: Comprehensive RT-STAMP-SLR Debug Block (~500ms)
            // ═══════════════════════════════════════════════════════════
            const _debugNow = performance.now();
            if (_debugNow - _lastHandDebugLog >= 500) {
                _lastHandDebugLog = _debugNow;
                const streamActive = !!(mediaStream && mediaStream.active);

                // Raw MediaPipe result inspection
                const rawHandCount = (handResults && handResults.landmarks) ? handResults.landmarks.length : 0;
                const rawLmCount = (rawHandCount > 0 && handResults.landmarks[0]) ? handResults.landmarks[0].length : 0;
                let rawHandedness = 'UNKNOWN';
                let rawDetectionConf = 0;
                if (rawHandCount > 0 && handResults.handedness && handResults.handedness[0] && handResults.handedness[0][0]) {
                    rawHandedness = handResults.handedness[0][0].categoryName || 'UNKNOWN';
                    rawDetectionConf = handResults.handedness[0][0].score || 0;
                }

                const gestureWord = (hasHand && currentPrediction.word && currentPrediction.word !== '--')
                    ? currentPrediction.word.replace('_', ' ').toUpperCase()
                    : '--';
                const gestureConf = (currentPrediction.confidence)
                    ? `${(currentPrediction.confidence * 100).toFixed(1)}%`
                    : '0.0%';
                const gestureState = currentEarlyDecision.state || (hasHand ? 'COLLECTING' : 'NO_HAND');

                console.log(
                    `\n[RT-STAMP-SLR DEBUG]\n` +
                    `Frame: ${frameId}\n` +
                    `FPS: ${webcamFps}\n` +
                    `\n` +
                    `Camera: ${streamActive ? 'STREAM ACTIVE' : 'STREAM INACTIVE'}\n` +
                    `Video: ${videoEl.videoWidth}x${videoEl.videoHeight} readyState=${videoEl.readyState}\n` +
                    `Video currentTime: ${videoEl.currentTime.toFixed(2)}s\n` +
                    `Canvas: ${canvasEl.width}x${canvasEl.height}\n` +
                    `\n` +
                    `MediaPipe: ${mediaPipeReady ? 'INITIALIZED' : 'NOT READY'}\n` +
                    `detectForVideo called: YES (calls=${_detectCallCount}, errors=${_detectErrorCount})\n` +
                    `detectForVideo error: ${handDetectError ? handDetectError.message : 'NONE'}\n` +
                    `Timestamp used: ${videoTimestampMs}\n` +
                    `\n` +
                    `[HAND DETECTION DEBUG]\n` +
                    `MediaPipe result exists: ${handResults ? 'YES' : 'NO (null)'}\n` +
                    `Raw hands detected: ${rawHandCount}\n` +
                    `Raw landmark count: ${rawLmCount}\n` +
                    `Handedness: ${rawHandedness}\n` +
                    `Detection confidence: ${rawDetectionConf.toFixed(4)}\n` +
                    `\n` +
                    `Hand (processed): ${hasHand ? 'YES' : 'NO'}\n` +
                    `Landmarks (processed): ${lmCount}\n` +
                    `\n` +
                    `Token: ${hasHand ? 'YES' : 'NO'}\n` +
                    `Prediction: ${gestureWord}\n` +
                    `Confidence: ${gestureConf}\n` +
                    `State: ${gestureState}\n` +
                    `\n` +
                    `Timing:\n` +
                    `  Hand MP: ${mpHandInferenceMs.toFixed(1)}ms\n` +
                    `  Render: ${renderDurationMs.toFixed(1)}ms\n` +
                    `  Token calc: ${tokenCalculationMs.toFixed(1)}ms\n` +
                    `  Backend RTT: ${roundTripLatencyMs}ms`
                );

                if (hasHand && landmarkData.hands[0]) {
                    const h0 = landmarkData.hands[0];
                    const w0 = h0[0];
                    const i8 = h0[8];
                    const vw = videoEl.videoWidth;
                    const vh = videoEl.videoHeight;
                    const cw = canvasEl.width;
                    const ch = canvasEl.height;
                    const w0Z = (w0 && w0.length >= 3 && w0[2] !== undefined && w0[2] !== null) ? `, ${w0[2].toFixed(4)}` : '';
                    const i8Z = (i8 && i8.length >= 3 && i8[2] !== undefined && i8[2] !== null) ? `, ${i8[2].toFixed(4)}` : '';
                    console.log(
                        `[LANDMARK COORD VERIFICATION]\n` +
                        `WRIST #0: MP (${w0[0].toFixed(4)}, ${w0[1].toFixed(4)}${w0Z}) | Video: ${vw}x${vh} | Canvas: ${cw}x${ch} | Draw: (${(w0[0] * cw).toFixed(1)}, ${(w0[1] * ch).toFixed(1)})\n` +
                        `INDEX TIP #8: MP (${i8[0].toFixed(4)}, ${i8[1].toFixed(4)}${i8Z}) | Video: ${vw}x${vh} | Canvas: ${cw}x${ch} | Draw: (${(i8[0] * cw).toFixed(1)}, ${(i8[1] * ch).toFixed(1)})`
                    );
                }
            }

            // Compute 6D token and send for backend CNN-GRU inference (async, non-blocking)
            sendTokenToBackend(landmarkData, frameId);
        } else {
            updateDebugHUDLiveMetrics(false, 0, 0, null);
        }

    } catch (loopError) {
        // PHASE 18: Log but do NOT let the error kill the rAF loop
        console.error('[PROCESSING LOOP EXCEPTION]', loopError);
    }

    if (isWebcamRunning) {
        requestAnimationFrame(processWebcamLoop);
    }
}

// ─── Live Debug HUD DOM Updaters (Phase 3) ────────────────────────
function updateDebugHUDLiveMetrics(hasHand, handCount, lmCount, landmarkData = null) {
    const streamActive = !!(mediaStream && mediaStream.active);
    const videoTrack = mediaStream && mediaStream.getVideoTracks().length > 0 ? mediaStream.getVideoTracks()[0] : null;
    const trackState = videoTrack ? videoTrack.readyState.toUpperCase() : 'NONE';

    if (dbgCameraState) {
        dbgCameraState.textContent = currentCameraState;
        dbgCameraState.className = 'dbg-val ' + (
            currentCameraState === 'CAMERA_LIVE' ? 'val-active' :
            currentCameraState === 'CAMERA_STARTING' ? 'val-warning' :
            currentCameraState === 'CAMERA_ERROR' ? 'val-error' : 'val-inactive'
        );
    }

    if (dbgCameraStream) {
        dbgCameraStream.textContent = streamActive ? 'STREAM ACTIVE' : 'STREAM INACTIVE';
        dbgCameraStream.className = 'dbg-val ' + (streamActive ? 'val-active' : 'val-inactive');
    }

    if (dbgReadyState) {
        dbgReadyState.textContent = videoEl ? videoEl.readyState : '0';
    }

    if (dbgDimensions) {
        dbgDimensions.textContent = videoEl && videoEl.videoWidth > 0
            ? `${videoEl.videoWidth} x ${videoEl.videoHeight}`
            : '0 x 0';
    }

    const dbgCanvasDimensions = document.getElementById('dbgCanvasDimensions');
    if (dbgCanvasDimensions) {
        dbgCanvasDimensions.textContent = canvasEl && canvasEl.width > 0
            ? `${canvasEl.width} x ${canvasEl.height}`
            : '0 x 0';
    }

    if (dbgCurrentTime) {
        dbgCurrentTime.textContent = videoEl ? `${videoEl.currentTime.toFixed(2)}s` : '0.00s';
    }

    if (dbgVideoPaused) {
        dbgVideoPaused.textContent = videoEl ? String(videoEl.paused) : 'true';
    }

    if (dbgVideoTrack) {
        dbgVideoTrack.textContent = trackState;
    }

    if (dbgPipelineState) {
        dbgPipelineState.textContent = currentPipelineState;
        dbgPipelineState.className = 'dbg-val ' + (
            currentPipelineState === 'PIPELINE_READY' ? 'val-active' :
            currentPipelineState === 'PIPELINE_ERROR' ? 'val-error' : 'val-warning'
        );
    }

    if (dbgLoopStatus) {
        dbgLoopStatus.textContent = isWebcamRunning ? 'RUNNING' : 'STOPPED';
        dbgLoopStatus.className = 'dbg-val ' + (isWebcamRunning ? 'val-running' : 'val-stopped');
    }

    if (dbgFrameId) {
        dbgFrameId.textContent = frameId;
    }

    if (dbgDetectorStatus) {
        if (mediaPipeReady) {
            dbgDetectorStatus.textContent = handLandmarker ? 'READY (Browser MP)' : 'READY';
            dbgDetectorStatus.className = 'dbg-val val-active';
        } else if (currentPipelineState === 'PIPELINE_ERROR') {
            dbgDetectorStatus.textContent = _lastDetectError ? `ERROR (${_lastDetectError.name || 'InitFailed'})` : 'ERROR';
            dbgDetectorStatus.className = 'dbg-val val-error';
        } else {
            dbgDetectorStatus.textContent = 'STARTING';
            dbgDetectorStatus.className = 'dbg-val val-warning';
        }
    }
    if (dbgHandDetected) {
        if (!hasHand) {
            dbgHandDetected.textContent = 'NO';
            dbgHandDetected.className = 'dbg-val val-no';
        } else if (landmarkData && landmarkData.left_hand && landmarkData.right_hand) {
            dbgHandDetected.textContent = '2 HANDS (L + R)';
            dbgHandDetected.className = 'dbg-val val-yes';
        } else if (landmarkData && landmarkData.left_hand) {
            dbgHandDetected.textContent = '1 HAND (LEFT)';
            dbgHandDetected.className = 'dbg-val val-yes';
        } else if (landmarkData && landmarkData.right_hand) {
            dbgHandDetected.textContent = '1 HAND (RIGHT)';
            dbgHandDetected.className = 'dbg-val val-yes';
        } else {
            dbgHandDetected.textContent = handCount > 1 ? `YES (${handCount} Hands)` : 'YES';
            dbgHandDetected.className = 'dbg-val val-yes';
        }
    }
    if (dbgLandmarkCount) {
        if (!hasHand) {
            dbgLandmarkCount.textContent = '0';
        } else if (landmarkData && landmarkData.left_hand && landmarkData.right_hand) {
            dbgLandmarkCount.textContent = `${lmCount} (21 L + 21 R)`;
        } else if (landmarkData && landmarkData.left_hand) {
            dbgLandmarkCount.textContent = '21 (Left)';
        } else if (landmarkData && landmarkData.right_hand) {
            dbgLandmarkCount.textContent = '21 (Right)';
        } else {
            dbgLandmarkCount.textContent = lmCount;
        }
    }
    if (dbgFpsPill) {
        dbgFpsPill.textContent = `${webcamFps} FPS`;
    }

    if (valMpHandTime) valMpHandTime.textContent = `${mpHandInferenceMs.toFixed(1)}ms`;
    if (valMpPoseTime) valMpPoseTime.textContent = `${mpPoseInferenceMs.toFixed(1)}ms`;
    if (valRenderTime) valRenderTime.textContent = `${renderDurationMs.toFixed(1)}ms`;
    if (valTokenCalcTime) valTokenCalcTime.textContent = `${tokenCalculationMs.toFixed(1)}ms`;
    if (valBackendRtt) valBackendRtt.textContent = `${roundTripLatencyMs}ms`;
}

// Adaptive dual-hand landmark smoothing & missed-frame tolerance state
let smoothedLeftHand = null;
let smoothedRightHand = null;
let missedLeftFramesCount = 0;
let missedRightFramesCount = 0;
const MAX_MISSED_HAND_FRAMES = 2; // Tolerance of 2 missed frames (~50ms) to prevent flicker

function smoothSingleHandPoints(rawPoints, prevSmoothed) {
    if (!prevSmoothed || !Array.isArray(prevSmoothed) || prevSmoothed.length !== rawPoints.length) {
        return rawPoints;
    }
    return rawPoints.map((raw, idx) => {
        const prev = prevSmoothed[idx] || raw;
        const dx = raw[0] - prev[0];
        const dy = raw[1] - prev[1];
        const distSq = dx * dx + dy * dy;
        const alpha = distSq > 0.0001 ? 0.80 : 0.50;
        return [
            prev[0] + alpha * (raw[0] - prev[0]),
            prev[1] + alpha * (raw[1] - prev[1]),
            prev[2] + alpha * (raw[2] - prev[2])
        ];
    });
}

function calculateHandCenter(handPoints) {
    if (!handPoints || handPoints.length === 0) return null;
    let sumX = 0, sumY = 0;
    handPoints.forEach(pt => {
        sumX += pt[0];
        sumY += pt[1];
    });
    return [sumX / handPoints.length, sumY / handPoints.length];
}

// ─── Build Landmark Data (Hand Tracking & Selection matching Python LandmarkExtractor) ───
function buildLandmarkData(handResults, poseResults) {
    const data = {
        pose: {},
        hands: [],
        hand_center: [0.5, 0.5],
        shoulder_center: [0.5, 0.35],
        hand_count: 0,
        is_fallback: false
    };

    // Extract pose landmarks (LS #11, RS #12)
    if (poseResults && poseResults.landmarks && poseResults.landmarks.length > 0) {
        const lms = poseResults.landmarks[0];
        if (lms.length > 16) {
            data.pose = {
                LW: [lms[15].x, lms[15].y, lms[15].z],
                RW: [lms[16].x, lms[16].y, lms[16].z],
                LE: [lms[13].x, lms[13].y, lms[13].z],
                RE: [lms[14].x, lms[14].y, lms[14].z],
                LS: [lms[11].x, lms[11].y, lms[11].z],
                RS: [lms[12].x, lms[12].y, lms[12].z]
            };
            data.shoulder_center = [
                (lms[11].x + lms[12].x) / 2,
                (lms[11].y + lms[12].y) / 2
            ];
        }
    }

    // Extract all raw hand landmark arrays (ignoring unstable MediaPipe handedness labels)
    const detectedHandsList = [];
    if (handResults && handResults.landmarks && handResults.landmarks.length > 0) {
        handResults.landmarks.forEach((hand) => {
            if (hand && hand.length === 21) {
                const pts = hand.map(lm => [lm.x, lm.y, lm.z]);
                detectedHandsList.push(pts);
            }
        });
    }

    data.hands = detectedHandsList;
    data.hand_count = detectedHandsList.length;

    // Track hand center matching Python LandmarkExtractor:
    // If 1 hand -> use that hand's center
    // If >1 hands -> choose hand closest to prevHandCenter (if set), else top-most hand (lowest Y)
    if (detectedHandsList.length > 0) {
        const centers = detectedHandsList.map(h => calculateHandCenter(h));
        let selectedCenter = null;
        if (centers.length === 1) {
            selectedCenter = centers[0];
        } else if (prevHandCenter !== null) {
            const px = prevHandCenter[0], py = prevHandCenter[1];
            let bestIdx = 0;
            let minDistSq = Infinity;
            centers.forEach((c, idx) => {
                const dSq = (c[0] - px) ** 2 + (c[1] - py) ** 2;
                if (dSq < minDistSq) {
                    minDistSq = dSq;
                    bestIdx = idx;
                }
            });
            selectedCenter = centers[bestIdx];
        } else {
            let bestIdx = 0;
            let minY = Infinity;
            centers.forEach((c, idx) => {
                if (c[1] < minY) {
                    minY = c[1];
                    bestIdx = idx;
                }
            });
            selectedCenter = centers[bestIdx];
        }
        data.hand_center = selectedCenter;
    }

    return data;
}

// ─── 6D Token Computation (Browser-Side) ─────────────────────────
let prevHandCenter = null;
let currentSequenceId = 1;
let consecutiveNoHandFrames = 0;
const CONSECUTIVE_NO_HAND_FOR_RESET = 4; // Require 4 consecutive frames (~100-150ms) of no-hand before sequence reset

function resetTokenizerState() {
    prevHandCenter = null;
    currentSequenceId++;
    consecutiveNoHandFrames = 0;
}

function compute6DToken(landmarkData) {
    const hasHand = landmarkData && landmarkData.hands && landmarkData.hands.length > 0;
    if (!hasHand) {
        consecutiveNoHandFrames++;
        if (consecutiveNoHandFrames >= CONSECUTIVE_NO_HAND_FOR_RESET) {
            resetTokenizerState();
        }
        return null; // Return null when no hand detected
    }

    consecutiveNoHandFrames = 0;

    const hx = (landmarkData.hand_center && landmarkData.hand_center[0] !== undefined) ? landmarkData.hand_center[0] : 0.5;
    const hy = (landmarkData.hand_center && landmarkData.hand_center[1] !== undefined) ? landmarkData.hand_center[1] : 0.5;

    const sx = (landmarkData.shoulder_center && landmarkData.shoulder_center[0] !== undefined) ? landmarkData.shoulder_center[0] : 0.5;
    const sy = (landmarkData.shoulder_center && landmarkData.shoulder_center[1] !== undefined) ? landmarkData.shoulder_center[1] : 0.35;

    // Hand velocity delta calculation (resets on first frame of gesture to avoid artificial jump)
    let mx = 0.0, my = 0.0;
    if (prevHandCenter !== null) {
        const jumpDist = Math.hypot(hx - prevHandCenter[0], hy - prevHandCenter[1]);
        if (jumpDist > 0.20) {
            mx = 0.0;
            my = 0.0;
        } else {
            mx = hx - prevHandCenter[0];
            my = hy - prevHandCenter[1];
        }
    } else {
        // First valid hand frame after reset — initialize position without motion jump
        mx = 0.0;
        my = 0.0;
    }
    prevHandCenter = [hx, hy];

    // Relative to shoulder baseline
    const rx = hx - sx;
    const ry = hy - sy;

    const token = [hx, hy, mx, my, rx, ry];
    updateTokenDisplay(token, true);
    return token;
}

function updateTokenDisplay(token, hasHand) {
    if (!hasHand || !token || token.length < 6) {
        if (tkHx) tkHx.textContent = '--';
        if (tkHy) tkHy.textContent = '--';
        if (tkMx) tkMx.textContent = '--';
        if (tkMy) tkMy.textContent = '--';
        if (tkRx) tkRx.textContent = '--';
        if (tkRy) tkRy.textContent = '--';
        return;
    }
    if (tkHx) tkHx.textContent = Number(token[0]).toFixed(2);
    if (tkHy) tkHy.textContent = Number(token[1]).toFixed(2);
    if (tkMx) tkMx.textContent = Number(token[2]).toFixed(2);
    if (tkMy) tkMy.textContent = Number(token[3]).toFixed(2);
    if (tkRx) tkRx.textContent = Number(token[4]).toFixed(2);
    if (tkRy) tkRy.textContent = Number(token[5]).toFixed(2);
}

// ─── Send Token to Backend ───────────────────────────────────────
let lastTokenSentTimestamp = 0;
let lastSentHadHand = false;
let _lastWebFrameLog = 0;
const TOKEN_SEND_INTERVAL_MS = 50; // ~20 FPS target for optimal backend synchronization

async function sendTokenToBackend(landmarkData, currentFrameId) {
    // Single in-flight request guard: serialize token requests to prevent overlap
    if (isProcessingToken) {
        droppedFramesCount++;
        return;
    }

    const hasHand = landmarkData && landmarkData.hands && landmarkData.hands.length > 0;
    const now = performance.now();

    // Handle gesture transition / hand reset
    if (!hasHand) {
        if (!lastSentHadHand) {
            return; // Backend already notified of NO_HAND state
        }
    } else {
        if (now - lastTokenSentTimestamp < TOKEN_SEND_INTERVAL_MS) {
            return;
        }
    }

    lastTokenSentTimestamp = now;
    lastSentHadHand = hasHand;
    isProcessingToken = true;
    const token = compute6DToken(landmarkData);
    const captureTime = now;

    if (now - _lastWebFrameLog >= 1000) {
        _lastWebFrameLog = now;
        console.log(`[WEB FRAME] Sending frame to /api/process_token (frame_id=${currentFrameId}, seq_id=${currentSequenceId}, hasHand=${hasHand})`);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    try {
        const res = await fetch('/api/process_token', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({
                token: token,
                frame_id: currentFrameId,
                sequence_id: currentSequenceId,
                timestamp_ms: captureTime,
                pose: landmarkData.pose,
                hand_center: landmarkData.hand_center,
                shoulder_center: landmarkData.shoulder_center,
                has_hand: hasHand,
                hand_count: landmarkData.hand_count || (hasHand ? 1 : 0),
                primary_hand: "TrackedHand"
            })
        });
        clearTimeout(timeoutId);

        if (!res.ok) {
            throw new Error(`Server returned HTTP ${res.status}`);
        }

        const data = await res.json();
        processedFramesCount++;
        roundTripLatencyMs = Math.round(performance.now() - captureTime);

        if (!hasHand) {
            currentPrediction = { word: "--", confidence: 0.0 };
            latestDebugMetrics.bufferStatus = "0/25";
            currentEarlyDecision = {
                state: "NO_HAND",
                sustained_count: 0,
                sustained_target: 3,
                last_accepted: currentEarlyDecision.last_accepted || "--"
            };
            // Clear stale prediction display on NO_HAND
            valDetectedWord.textContent = '--';
            valConfidencePct.textContent = '0%';
            confidenceBarFill.style.width = '0%';
        }

        updateUI(data, hasHand);

        if (data.buffer_status) {
            latestDebugMetrics.bufferStatus = data.buffer_status;
        }
    } catch (e) {
        clearTimeout(timeoutId);
        if (e.name === 'AbortError') {
            console.warn('[Token Process] Request timed out (2.5s). Resetting lock.');
        } else {
            console.error('Token process error:', e);
        }
    } finally {
        isProcessingToken = false;
    }
}

// ─── Draw Real Landmarks with Correct Hand Skeleton (Phase 7) ────
function drawRealLandmarks(landmarkData) {
    // Clear canvas to 100% transparent — DO NOT draw any opaque box over the video!
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    const w = canvasEl.width;
    const h = canvasEl.height;

    // Draw Pose joints and arm lines
    if (landmarkData.pose) {
        const p = landmarkData.pose;

        // Shoulder connector
        if (p.LS && p.RS) {
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(p.LS[0] * w, p.LS[1] * h);
            ctx.lineTo(p.RS[0] * w, p.RS[1] * h);
            ctx.stroke();
        }

        // Arm bones: Shoulder → Elbow → Wrist
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        if (p.LS && p.LE) {
            ctx.beginPath();
            ctx.moveTo(p.LS[0] * w, p.LS[1] * h);
            ctx.lineTo(p.LE[0] * w, p.LE[1] * h);
            ctx.stroke();
        }
        if (p.LE && p.LW) {
            ctx.beginPath();
            ctx.moveTo(p.LE[0] * w, p.LE[1] * h);
            ctx.lineTo(p.LW[0] * w, p.LW[1] * h);
            ctx.stroke();
        }
        if (p.RS && p.RE) {
            ctx.beginPath();
            ctx.moveTo(p.RS[0] * w, p.RS[1] * h);
            ctx.lineTo(p.RE[0] * w, p.RE[1] * h);
            ctx.stroke();
        }
        if (p.RE && p.RW) {
            ctx.beginPath();
            ctx.moveTo(p.RE[0] * w, p.RE[1] * h);
            ctx.lineTo(p.RW[0] * w, p.RW[1] * h);
            ctx.stroke();
        }

        // Pose joint dots
        ctx.fillStyle = '#06b6d4';
        ['LS', 'RS', 'LE', 'RE', 'LW', 'RW'].forEach(k => {
            if (p[k]) {
                ctx.beginPath();
                ctx.arc(p[k][0] * w, p[k][1] * h, 5, 0, 2 * Math.PI);
                ctx.fill();
            }
        });
    }

    // Draw Dual Hand Skeletons (up to 2 hands, 21 landmarks & 20 bones each)
    if (landmarkData.hands && landmarkData.hands.length > 0) {
        landmarkData.hands.forEach((hand, hIdx) => {
            if (hand.length >= 21) {
                // Hand 0 (Pink/Rose #f43f5e), Hand 1 (Cyan/Sky #06b6d4)
                ctx.strokeStyle = hIdx === 0 ? '#f43f5e' : '#06b6d4';
                ctx.lineWidth = 2.5;
                ctx.lineCap = 'round';
                for (const [from, to] of HAND_CONNECTIONS) {
                    if (hand[from] && hand[to]) {
                        ctx.beginPath();
                        ctx.moveTo(hand[from][0] * w, hand[from][1] * h);
                        ctx.lineTo(hand[to][0] * w, hand[to][1] * h);
                        ctx.stroke();
                    }
                }

                // Draw joint dots
                ctx.fillStyle = hIdx === 0 ? '#ec4899' : '#38bdf8';
                hand.forEach((pt, idx) => {
                    const isTip = FINGERTIP_INDICES.includes(idx);
                    const radius = isTip ? 6 : 3;
                    ctx.beginPath();
                    ctx.arc(pt[0] * w, pt[1] * h, radius, 0, 2 * Math.PI);
                    ctx.fill();
                });

                // Highlight fingertips (#4 Thumb, #8 Index, #12 Middle, #16 Ring, #20 Pinky)
                ctx.fillStyle = '#fbbf24';
                for (const tipIdx of FINGERTIP_INDICES) {
                    if (hand[tipIdx]) {
                        ctx.beginPath();
                        ctx.arc(hand[tipIdx][0] * w, hand[tipIdx][1] * h, 5.5, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                }
            }
        });
    }
}

// ─── Update UI Widgets (Phase 10 & 11) ────────────────────────────
function updateUI(data, hasHand) {
    // ── [GESTURE DEBUG] Mirror backend diagnostic block in browser console ──
    // Only log when a full 25-token prediction is available (buffer_status === "25/25")
    if (data.buffer_status === '25/25') {
        const _now = performance.now();
        if (_now - _lastNoWebLog >= 500) {
            _lastNoWebLog = _now;
            console.log(
                `\n[GESTURE DEBUG]\n` +
                `21CLASS        = ${data.primary_class}\n` +
                `21CLASS_CONF   = ${(data.primary_confidence || 0).toFixed(4)}\n` +
                `NO_PROBABILITY = ${(data.no_probability !== undefined ? data.no_probability : (data.binary_no ? data.binary_no.no_probability : 0)).toFixed(4)}\n` +
                `NO_CONFIRMATIONS = ${data.no_confirmations !== undefined ? data.no_confirmations : (data.binary_no ? data.binary_no.consecutive_count : 0)}\n` +
                `NO_CONFIRMED   = ${data.no_confirmed !== undefined ? data.no_confirmed : (data.binary_no ? data.binary_no.no_confirmed : false)}\n` +
                `FINAL_CLASS    = ${data.final_class}`
            );
            // Also log token coordinates for NO-token coordinate verification
            if (data.token && data.token.length >= 6) {
                console.log(
                    `[NO TOKEN]\n` +
                    `Hx=${data.token[0].toFixed(4)}\n` +
                    `Hy=${data.token[1].toFixed(4)}\n` +
                    `Mx=${data.token[2].toFixed(4)}\n` +
                    `My=${data.token[3].toFixed(4)}\n` +
                    `Rx=${data.token[4].toFixed(4)}\n` +
                    `Ry=${data.token[5].toFixed(4)}`
                );
            }
        }
    } else if (data.binary_no) {
        // Still log binary_no for partial buffers at lower rate
        const _now = performance.now();
        if (_now - _lastNoWebLog >= 1000) {
            _lastNoWebLog = _now;
            console.log(
                `[NO WEB] prob=${data.binary_no.no_probability.toFixed(4)} | ` +
                `pred=${data.binary_no.no_prediction} | ` +
                `confirmed=${data.binary_no.no_confirmed} | ` +
                `count=${data.binary_no.consecutive_count}`
            );
        }
    }

    // Motion Energy & Threshold
    if (data.motion_energy !== undefined) {
        valMotionEnergy.textContent = data.motion_energy.toFixed(4);
    }
    if (data.threshold !== undefined) {
        valThreshold.textContent = data.threshold.toFixed(4);

        motionHistory.shift();
        motionHistory.push(data.motion_energy);
        thresholdHistory.shift();
        thresholdHistory.push(data.threshold);
        drawMotionGraph();
    }

    // 6D Token Values (Real values from backend / tokenizer)
    updateTokenDisplay(data.token, hasHand);

    // Active Word Extraction for Immediate UI Synchronization
    const activeWordClean = (data.prediction && data.prediction.word && data.prediction.word !== 'BUFFERING' && data.prediction.word !== 'COLLECTING GESTURE...' && data.prediction.word !== '--' && (data.prediction.confidence || 0) >= 0.40)
        ? data.prediction.word.replace('_', ' ').toUpperCase()
        : null;

    // Prediction Confidence Bar & Word (Step 7: directly display backend final_class and confidence)
    if (data.prediction || data.final_class) {
        const rawWord = data.final_class || (data.prediction ? data.prediction.word : '--');
        const isNoFinal = (rawWord === 'no' || data.no_confirmed);
        const conf = isNoFinal
            ? (data.no_probability !== undefined ? data.no_probability : (data.binary_no ? data.binary_no.no_probability : 0.90))
            : ((data.primary_confidence !== undefined) ? data.primary_confidence : ((data.prediction && data.prediction.confidence) || 0));
        const confPct = Math.round(conf * 100);
        currentPrediction = { word: rawWord, confidence: conf };

        let displayWord = '--';
        if (!hasHand) {
            displayWord = '--';
            valConfidencePct.textContent = '0%';
            confidenceBarFill.style.width = '0%';
        } else if (rawWord === 'COLLECTING GESTURE...') {
            displayWord = 'COLLECTING GESTURE...';
            valConfidencePct.textContent = '0%';
            confidenceBarFill.style.width = '0%';
        } else if (rawWord && rawWord !== '--' && rawWord !== 'BUFFERING' && rawWord !== 'WAITING FOR CLEAR GESTURE') {
            displayWord = rawWord.replace('_', ' ').toUpperCase();
            valConfidencePct.textContent = `${confPct}%`;
            confidenceBarFill.style.width = `${confPct}%`;
        } else {
            displayWord = '--';
            valConfidencePct.textContent = `${confPct}%`;
            confidenceBarFill.style.width = `${confPct}%`;
        }

        valDetectedWord.textContent = displayWord;

        // Update Single Gesture Test Card (Phase 10)
        if (sgValSign) sgValSign.textContent = displayWord;
        if (sgValConf) sgValConf.textContent = hasHand ? `${confPct}%` : '0.0%';
    }

    // Early Decision State Transitions (Phase 11)
    if (data.early_decision) {
        currentEarlyDecision = data.early_decision;
        let determinedState = data.early_decision.state;

        if (!isWebcamRunning || currentCameraState === 'CAMERA_OFF') {
            determinedState = 'CAMERA_OFF';
        } else if (!hasHand) {
            determinedState = 'NO_HAND';
        } else if (data.early_decision.accepted || determinedState === 'LOCKED' || determinedState === 'CONFIRMED') {
            determinedState = 'ACCEPTED';
        } else if (data.early_decision.state === 'COOLDOWN' || (data.buffer_status && data.buffer_status.includes('Cooldown'))) {
            determinedState = 'COOLDOWN';
        } else if (data.prediction && data.prediction.word === 'COLLECTING GESTURE...') {
            determinedState = 'COLLECTING';
        } else if (data.buffer_status && data.buffer_status.includes('Idle')) {
            determinedState = 'READY';
        } else if (data.prediction && data.prediction.confidence > 0.3) {
            determinedState = 'PREDICTING';
        } else {
            determinedState = 'READY';
        }

        updateSystemState(determinedState);
    }

    // Translation Output & Word Buffer Synchronization
    if (data.translation || data.final_class) {
        let text = (data.translation && data.translation.display_text && data.translation.display_text.trim() !== '')
            ? data.translation.display_text
            : '';

        const activeSign = data.final_class || (data.prediction ? data.prediction.word : null);

        // If no confirmed sentence string yet, format active sign according to language mode
        if (!text || text.trim() === '') {
            if (!isWebcamRunning || currentCameraState === 'CAMERA_OFF') {
                text = '<em>WAITING FOR WEBCAM</em>';
            } else if (!hasHand) {
                text = '<em>WAITING FOR HAND GESTURE</em>';
            } else if (activeSign === 'COLLECTING GESTURE...' || activeSign === 'BUFFERING') {
                text = '<em>COLLECTING GESTURE...</em>';
            } else if (activeSign && activeSign !== '--' && activeSign !== 'WAITING FOR CLEAR GESTURE') {
                if (currentLanguage === 'tamil') {
                    if (activeSign === 'water') text = 'தண்ணீர்';
                    else if (activeSign === 'no') text = 'இல்லை';
                    else if (activeSign === 'please') text = 'தயவுசெய்து (Thayavuseythu)';
                    else if (activeSign === 'school') text = 'பள்ளி (Palli)';
                    else text = (data.translation && data.translation.tamil) ? data.translation.tamil : activeSign;
                } else {
                    if (activeSign === 'water') text = 'Water (தண்ணீர்)';
                    else if (activeSign === 'no') text = 'No (இல்லை)';
                    else if (activeSign === 'please') text = 'Please.';
                    else if (activeSign === 'school') text = 'School';
                    else text = activeSign.charAt(0).toUpperCase() + activeSign.slice(1).replace('_', ' ') + '.';
                }
            } else {
                text = '<em>WAITING FOR HAND GESTURE</em>';
            }
        }

        translatedText.innerHTML = text;
        lastTranslationText = text;

        // Synchronize Word Buffer (confirmed words or active confirmed sign)
        let rawWords = (data.translation && data.translation.raw_words && data.translation.raw_words.length > 0)
            ? data.translation.raw_words
            : [];
        if (rawWords.length === 0 && activeSign && activeSign !== 'COLLECTING GESTURE...' && activeSign !== 'BUFFERING' && activeSign !== '--' && activeSign !== 'WAITING FOR CLEAR GESTURE' && hasHand) {
            rawWords = [activeSign];
        }
        updateWordBuffer(rawWords);
    }
}


function updateWordBuffer(words) {
    if (!wordBufferContainer) return;
    wordBufferContainer.innerHTML = '<span class="buffer-label">Word Buffer:</span>';

    if (!words || words.length === 0) {
        const emptyTag = document.createElement('span');
        emptyTag.className = 'buffer-empty';
        emptyTag.id = 'bufferEmpty';
        emptyTag.textContent = '--';
        wordBufferContainer.appendChild(emptyTag);
        return;
    }

    words.forEach(w => {
        const tag = document.createElement('span');
        tag.className = 'word-tag';
        tag.textContent = w.replace('_', ' ').toUpperCase();
        wordBufferContainer.appendChild(tag);
    });
}

// ─── Draw Motion Graph ───────────────────────────────────────────
function drawMotionGraph() {
    graphCtx.clearRect(0, 0, motionGraphCanvas.width, motionGraphCanvas.height);
    const w = motionGraphCanvas.width;
    const h = motionGraphCanvas.height;
    const maxVal = 0.1;

    // Motion Energy Line
    graphCtx.strokeStyle = '#3b82f6';
    graphCtx.lineWidth = 2;
    graphCtx.beginPath();
    for (let i = 0; i < motionHistory.length; i++) {
        const x = (i / (motionHistory.length - 1)) * w;
        const y = h - (motionHistory[i] / maxVal) * h;
        if (i === 0) graphCtx.moveTo(x, y);
        else graphCtx.lineTo(x, y);
    }
    graphCtx.stroke();

    // Threshold Line (Amber dashed)
    graphCtx.strokeStyle = '#f59e0b';
    graphCtx.lineWidth = 1;
    graphCtx.setLineDash([4, 4]);
    graphCtx.beginPath();
    for (let i = 0; i < thresholdHistory.length; i++) {
        const x = (i / (thresholdHistory.length - 1)) * w;
        const y = h - (thresholdHistory[i] / maxVal) * h;
        if (i === 0) graphCtx.moveTo(x, y);
        else graphCtx.lineTo(x, y);
    }
    graphCtx.stroke();
    graphCtx.setLineDash([]);
}

// ─── Language Toggle ─────────────────────────────────────────────
// Expose to global scope for inline onclick handlers
window.setLanguage = async function setLanguage(lang) {
    currentLanguage = lang;
    document.getElementById('btnLangEng').classList.toggle('active', lang === 'english');
    document.getElementById('btnLangTam').classList.toggle('active', lang === 'tamil');
    currentLangLabel.textContent = lang === 'english' ? 'ENGLISH SENTENCE' : 'TAMIL TRANSLATION (தமிழ்)';

    try {
        const res = await fetch('/api/toggle_language', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: lang })
        });
        const data = await res.json();
        if (data.translation && data.translation.display_text && data.translation.display_text.trim() !== '') {
            translatedText.innerHTML = data.translation.display_text;
            lastTranslationText = data.translation.display_text;
        } else {
            translatedText.innerHTML = isWebcamRunning ? '<em>WAITING FOR HAND GESTURE</em>' : '<em>WAITING FOR WEBCAM</em>';
        }
    } catch (e) {
        console.error('Toggle language error:', e);
    }
};

// ─── Clear Sentence ──────────────────────────────────────────────
async function clearSentence() {
    try {
        const res = await fetch('/api/clear_sentence', { method: 'POST' });
        const data = await res.json();
        translatedText.innerHTML = isWebcamRunning ? '<em>WAITING FOR HAND GESTURE</em>' : '<em>WAITING FOR WEBCAM</em>';
        lastTranslationText = "";
        updateWordBuffer([]);
    } catch (e) {
        console.error('Clear sentence error:', e);
    }
}

// ─── TTS ─────────────────────────────────────────────────────────
function speakTranslation() {
    if (!lastTranslationText) return;
    const utterance = new SpeechSynthesisUtterance(lastTranslationText);
    utterance.lang = currentLanguage === 'english' ? 'en-US' : 'ta-IN';
    window.speechSynthesis.speak(utterance);
}

// ─── Simulate Sign ───────────────────────────────────────────────
async function simulateSign(word) {
    try {
        const res = await fetch('/api/simulate_sign', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: word })
        });
        const data = await res.json();
        if (data.last_frame_result) {
            updateUI(data.last_frame_result);
        }
    } catch (e) {
        console.error('Simulate sign error:', e);
    }
}

// ─── Train Model ─────────────────────────────────────────────────
async function trainModel() {
    btnTrainModel.disabled = true;
    btnTrainModel.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training...';
    try {
        const res = await fetch('/api/train', { method: 'POST' });
        const data = await res.json();
        alert(`Model trained successfully!\nBest Accuracy: ${(data.training_result.best_val_acc * 100).toFixed(1)}%`);
    } catch (e) {
        alert('Training failed: ' + e.message);
    } finally {
        btnTrainModel.disabled = false;
        btnTrainModel.innerHTML = '<i class="fa-solid fa-brain"></i> Train Model';
    }
}


// ═══════════════════════════════════════════════════════════════════
// ISL Dataset Explorer — HuggingFace API Integration
// ═══════════════════════════════════════════════════════════════════

let datasetCurrentOffset = 0;
const DATASET_PAGE_SIZE = 5;
let datasetTotalRows = 0;

/**
 * Fetches ISL dataset rows from the backend proxy and renders them.
 */
async function loadDatasetBrowser(offset, length) {
    if (offset === undefined) offset = datasetCurrentOffset;
    if (length === undefined) length = DATASET_PAGE_SIZE;

    const loadingEl = document.getElementById('datasetLoading');
    const errorEl = document.getElementById('datasetError');
    const gridEl = document.getElementById('datasetCardsGrid');
    const paginationEl = document.getElementById('datasetPagination');

    // Show loading, hide error and grid
    if (loadingEl) loadingEl.style.display = 'flex';
    if (errorEl) errorEl.style.display = 'none';
    if (gridEl) gridEl.innerHTML = '';

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    try {
        const res = await fetch(`/api/dataset_browser?offset=${offset}&length=${length}`, {
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();

        // Hide loading
        if (loadingEl) loadingEl.style.display = 'none';

        // Update state
        datasetCurrentOffset = offset;
        datasetTotalRows = data.total || 0;

        // Update total count badge
        const totalCountEl = document.getElementById('datasetTotalCount');
        if (totalCountEl) totalCountEl.textContent = datasetTotalRows.toLocaleString();

        // Render cards
        renderDatasetCards(data.rows, gridEl);

        // Update pagination
        updateDatasetPagination(offset, length, datasetTotalRows, data.has_more);

    } catch (e) {
        console.error('[Dataset Explorer] Load error:', e);
        if (loadingEl) loadingEl.style.display = 'none';
        if (errorEl) {
            errorEl.style.display = 'flex';
            const msgEl = document.getElementById('datasetErrorMsg');
            if (msgEl) msgEl.textContent = `Failed to load dataset: ${e.message}`;
        }
    }
}

// Expose to global scope for inline onclick in the retry button
window.loadDatasetBrowser = loadDatasetBrowser;

/**
 * Renders dataset rows as interactive cards inside the grid container.
 */
function renderDatasetCards(rows, gridEl) {
    if (!gridEl || !rows || rows.length === 0) {
        if (gridEl) gridEl.innerHTML = '<p style="text-align:center; color:#9ca3af; padding:20px;">No dataset entries found.</p>';
        return;
    }

    gridEl.innerHTML = '';

    rows.forEach((row) => {
        const card = document.createElement('div');
        card.className = 'ds-entry-card';

        // Determine dataset source CSS class
        const srcClass = row.dataset === 'CISLR' ? 'src-cislr'
            : row.dataset === 'INCLUDE' ? 'src-include'
            : 'src-default';

        // Quality score classification
        const qs = row.quality_score || 0;
        const qualityClass = qs >= 0.8 ? 'quality-high' : qs >= 0.5 ? 'quality-medium' : 'quality-low';
        const qualityLabel = qs >= 0.8 ? '★ High Quality' : qs >= 0.5 ? '◆ Medium' : '▼ Low';

        // Duplicate status pill class
        const dupClass = row.duplicate_status === 'unique' ? 'pill-unique' : '';
        const revClass = row.review_status === 'accepted' ? 'pill-accepted' : '';

        // Duration formatting
        const durationStr = row.duration ? `${row.duration.toFixed(1)}s` : '--';
        const fpsStr = row.fps ? `${row.fps}` : '--';

        card.innerHTML = `
            <span class="ds-row-idx">#${row.row_idx}</span>
            <div class="ds-word-title">${escapeHtml(row.word)}</div>
            <span class="ds-source-label ${srcClass}">${escapeHtml(row.dataset)}</span>

            <div class="ds-meta-grid">
                <div class="ds-meta-item">
                    <span class="ds-meta-label">Category</span>
                    <span class="ds-meta-value">${escapeHtml(row.signer || '--')}</span>
                </div>
                <div class="ds-meta-item">
                    <span class="ds-meta-label">Resolution</span>
                    <span class="ds-meta-value">${escapeHtml(row.resolution || '--')}</span>
                </div>
                <div class="ds-meta-item">
                    <span class="ds-meta-label">Duration</span>
                    <span class="ds-meta-value">${durationStr}</span>
                </div>
                <div class="ds-meta-item">
                    <span class="ds-meta-label">FPS</span>
                    <span class="ds-meta-value">${fpsStr}</span>
                </div>
            </div>

            <div class="ds-quality-badge ${qualityClass}">
                <i class="fa-solid fa-star"></i> Quality: ${(qs * 100).toFixed(0)}% ${qualityLabel}
            </div>

            <div class="ds-status-pills">
                <span class="ds-pill ${dupClass}">${escapeHtml(row.duplicate_status || 'unknown')}</span>
                <span class="ds-pill ${revClass}">${escapeHtml(row.review_status || 'pending')}</span>
            </div>

            ${row.repository ? `<a class="ds-video-link" href="${escapeHtml(row.repository)}" target="_blank" rel="noopener">
                <i class="fa-solid fa-film"></i> View Source Repository
            </a>` : ''}
        `;

        gridEl.appendChild(card);
    });
}

/**
 * Updates pagination button states and info label.
 */
function updateDatasetPagination(offset, length, total, hasMore) {
    const prevBtn = document.getElementById('btnDatasetPrev');
    const nextBtn = document.getElementById('btnDatasetNext');
    const infoEl = document.getElementById('paginationInfo');

    if (prevBtn) prevBtn.disabled = offset <= 0;
    if (nextBtn) nextBtn.disabled = !hasMore;

    const from = total > 0 ? offset + 1 : 0;
    const to = Math.min(offset + length, total);
    if (infoEl) infoEl.textContent = `Showing ${from}–${to} of ${total.toLocaleString()}`;
}

/**
 * Simple HTML escaper to prevent XSS from API data.
 */
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ─── Dataset Explorer Event Listeners & Auto-Load ───────────────
document.addEventListener('DOMContentLoaded', () => {
    // Pagination buttons
    const prevBtn = document.getElementById('btnDatasetPrev');
    const nextBtn = document.getElementById('btnDatasetNext');
    const refreshBtn = document.getElementById('btnRefreshDataset');

    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            const newOffset = Math.max(0, datasetCurrentOffset - DATASET_PAGE_SIZE);
            loadDatasetBrowser(newOffset, DATASET_PAGE_SIZE);
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            const newOffset = datasetCurrentOffset + DATASET_PAGE_SIZE;
            if (newOffset < datasetTotalRows) {
                loadDatasetBrowser(newOffset, DATASET_PAGE_SIZE);
            }
        });
    }

    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            refreshBtn.classList.add('btn-spin');
            loadDatasetBrowser(datasetCurrentOffset, DATASET_PAGE_SIZE).finally(() => {
                setTimeout(() => refreshBtn.classList.remove('btn-spin'), 600);
            });
        });
    }

    // Auto-load first page of dataset
    loadDatasetBrowser(0, DATASET_PAGE_SIZE);
});

