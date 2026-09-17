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

// ─── State Machine Transition Helper (Phase 11) ─────────────────
function updateSystemState(newState, reason) {
    currentSystemState = newState;
    if (decisionStateBadge) {
        decisionStateBadge.textContent = newState;
        decisionStateBadge.className = 'decision-badge';
        if (newState === 'CAMERA_OFF') decisionStateBadge.classList.add('state-camera-off');
        else if (newState === 'CAMERA_STARTING') decisionStateBadge.classList.add('state-camera-starting');
        else if (newState === 'CAMERA_ERROR') decisionStateBadge.classList.add('state-camera-error');
        else if (newState === 'NO_HAND') decisionStateBadge.classList.add('state-no-hand');
        else if (newState === 'COLLECTING') decisionStateBadge.classList.add('state-collecting');
        else if (newState === 'READY') decisionStateBadge.classList.add('state-ready');
        else if (newState === 'PREDICTING') decisionStateBadge.classList.add('state-predicting');
        else if (newState === 'ACCEPTED') decisionStateBadge.classList.add('state-accepted');
    }

    if (sgValState) {
        sgValState.textContent = newState;
        sgValState.className = 'sg-state-tag' + (newState === 'ACCEPTED' ? ' accepted' : '');
    }

    if (reason) {
        console.log(`[Pipeline State] ${newState}: ${reason}`);
    }
}

// ─── Init ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateSystemState('CAMERA_OFF', 'Application initialized, camera idle.');
    loadVocabulary();
    initEventListeners();
    drawMotionGraph();
    initMediaPipe();
});

function initEventListeners() {
    btnToggleWebcam.addEventListener('click', toggleWebcam);
    btnSimulateDemo.addEventListener('click', () => simulateSign('water'));
    btnSimulateSelected.addEventListener('click', () => {
        const word = selectVocab.value;
        if (word) simulateSign(word);
    });
    btnTrainModel.addEventListener('click', trainModel);
    btnTTS.addEventListener('click', speakTranslation);
    btnClear.addEventListener('click', clearSentence);

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

// ─── MediaPipe Browser-Side Initialization (Phase 6 Single-Hand) ─
async function initMediaPipe() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );

        // Configure SINGLE HAND mode: numHands = 1 with 0.40 confidence for reliable detection
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 2,
            minHandDetectionConfidence: 0.40,
            minHandPresenceConfidence: 0.40,
            minTrackingConfidence: 0.30
        });

        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.30,
            minPosePresenceConfidence: 0.30,
            minTrackingConfidence: 0.30
        });

        mediaPipeReady = true;
        if (dbgDetectorStatus) dbgDetectorStatus.textContent = 'READY (GPU)';
        console.log("[MediaPipe] Browser-side Single-Hand + Pose landmarkers initialized (GPU delegate).");
    } catch (e) {
        console.warn("[MediaPipe] GPU delegate failed, falling back to CPU:", e.message);
        try {
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
            );

            handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    delegate: "CPU"
                },
                runningMode: "VIDEO",
                numHands: 2,
                minHandDetectionConfidence: 0.40,
                minHandPresenceConfidence: 0.40,
                minTrackingConfidence: 0.30
            });

            poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                    delegate: "CPU"
                },
                runningMode: "VIDEO",
                numPoses: 1,
                minPoseDetectionConfidence: 0.30,
                minPosePresenceConfidence: 0.30,
                minTrackingConfidence: 0.30
            });

            mediaPipeReady = true;
            if (dbgDetectorStatus) dbgDetectorStatus.textContent = 'READY (CPU)';
            console.log("[MediaPipe] Browser-side Single-Hand landmarkers initialized (CPU delegate).");
        } catch (e2) {
            console.error("[MediaPipe] Failed to initialize:", e2);
            mediaPipeReady = false;
            if (dbgDetectorStatus) dbgDetectorStatus.textContent = 'INIT_FAILED';
            updateSystemState('CAMERA_ERROR', 'MediaPipe vision initialization failed');
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
        selectVocab.innerHTML = '';
        data.vocabulary.forEach(item => {
            const opt = document.createElement('option');
            opt.value = item.english;
            opt.textContent = `${item.id + 1}. ${item.display_english} → ${item.tamil}`;
            selectVocab.appendChild(opt);
        });
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

    // 1. Check MediaPipe readiness
    if (!mediaPipeReady) {
        updateSystemState('CAMERA_STARTING', 'MediaPipe models still loading...');
        let waited = 0;
        while (!mediaPipeReady && waited < 6000) {
            await new Promise(r => setTimeout(r, 250));
            waited += 250;
        }
        if (!mediaPipeReady) {
            updateSystemState('CAMERA_ERROR', 'MediaPipe failed to load in time');
            alert('MediaPipe is still initializing. Please check network connection and try again.');
            return;
        }
    }

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

        updateSystemState('NO_HAND', `Webcam active (${videoEl.videoWidth}x${videoEl.videoHeight}). Waiting for hand gesture.`);
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
    updateSystemState('CAMERA_OFF', 'Webcam stopped by user.');
    updateDebugHUDLiveMetrics(false, 0, 0, 0, false, 0);
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

            // Pose inference bypassed for Hand Landmarker performance optimization
            poseResults = null;
            mpPoseInferenceMs = 0;

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
                    console.log(
                        `[LANDMARK COORD VERIFICATION]\n` +
                        `WRIST #0: MP (${w0[0].toFixed(4)}, ${w0[1].toFixed(4)}) | Video: ${vw}x${vh} | Canvas: ${cw}x${ch} | Draw: (${(w0[0] * cw).toFixed(1)}, ${(w0[1] * ch).toFixed(1)})\n` +
                        `INDEX TIP #8: MP (${i8[0].toFixed(4)}, ${i8[1].toFixed(4)}) | Video: ${vw}x${vh} | Canvas: ${cw}x${ch} | Draw: (${(i8[0] * cw).toFixed(1)}, ${(i8[1] * ch).toFixed(1)})`
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
    if (dbgLoopStatus) {
        dbgLoopStatus.textContent = isWebcamRunning ? 'RUNNING' : 'STOPPED';
        dbgLoopStatus.className = 'dbg-val ' + (isWebcamRunning ? 'val-running' : 'val-stopped');
    }
    if (dbgFrameId) {
        dbgFrameId.textContent = frameId;
    }
    if (dbgDetectorStatus) {
        dbgDetectorStatus.textContent = mediaPipeReady ? 'RUNNING' : 'NOT RUNNING';
        dbgDetectorStatus.className = 'dbg-val ' + (mediaPipeReady ? 'val-active' : 'val-inactive');
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

// ─── Build Landmark Data (Two-Hand Detection & Left/Right Tracking) ───
function buildLandmarkData(handResults, poseResults) {
    const data = {
        pose: {},
        hands: [],
        left_hand: null,
        right_hand: null,
        left_hand_center: null,
        right_hand_center: null,
        hand_center: [0.5, 0.5],
        shoulder_center: [0.5, 0.35],
        hand_count: 0,
        is_fallback: false
    };

    // Extract pose landmarks (LW, RW, LE, RE, LS, RS)
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

    // Process up to 2 hands using MediaPipe handedness
    let rawLeft = null;
    let rawRight = null;

    if (handResults && handResults.landmarks && handResults.landmarks.length > 0) {
        const detectedHands = handResults.landmarks;
        const handednesses = handResults.handednesses || [];

        detectedHands.forEach((hand, idx) => {
            if (!hand || hand.length !== 21) return;
            const pts = hand.map(lm => [lm.x, lm.y, lm.z]);
            const handednessMeta = handednesses[idx] && handednesses[idx][0] ? handednesses[idx][0] : null;
            const label = handednessMeta ? handednessMeta.categoryName : null;

            if (label === 'Left') {
                rawLeft = pts;
            } else if (label === 'Right') {
                rawRight = pts;
            } else {
                // If handedness is unspecified, assign first to left or right based on X position
                if (!rawLeft && !rawRight) {
                    rawRight = pts;
                } else if (!rawLeft) {
                    rawLeft = pts;
                } else if (!rawRight) {
                    rawRight = pts;
                }
            }
        });
    }

    // Smooth Left Hand
    if (rawLeft) {
        smoothedLeftHand = smoothSingleHandPoints(rawLeft, smoothedLeftHand);
        missedLeftFramesCount = 0;
    } else {
        missedLeftFramesCount++;
        if (missedLeftFramesCount > MAX_MISSED_HAND_FRAMES) {
            smoothedLeftHand = null;
        }
    }

    // Smooth Right Hand
    if (rawRight) {
        smoothedRightHand = smoothSingleHandPoints(rawRight, smoothedRightHand);
        missedRightFramesCount = 0;
    } else {
        missedRightFramesCount++;
        if (missedRightFramesCount > MAX_MISSED_HAND_FRAMES) {
            smoothedRightHand = null;
        }
    }

    // Assemble hands array (Hand 0 = Left if present, Hand 1 = Right if present)
    if (smoothedLeftHand) {
        data.hands.push(smoothedLeftHand);
        data.left_hand = smoothedLeftHand;
        data.left_hand_center = calculateHandCenter(smoothedLeftHand);
    }
    if (smoothedRightHand) {
        data.hands.push(smoothedRightHand);
        data.right_hand = smoothedRightHand;
        data.right_hand_center = calculateHandCenter(smoothedRightHand);
    }

    data.hand_count = data.hands.length;

    // Primary dominant hand center for 6D CNN-GRU model (preserves uncorrupted gesture trajectory matching dataset training)
    if (data.left_hand_center && data.right_hand_center) {
        // When both hands are active, select primary active hand matching mirrored dataset training
        data.hand_center = data.left_hand_center;
    } else if (data.left_hand_center) {
        data.hand_center = data.left_hand_center;
    } else if (data.right_hand_center) {
        data.hand_center = data.right_hand_center;
    }

    return data;
}

// ─── 6D Token Computation (Browser-Side) ─────────────────────────
let prevHandCenter = null;

function compute6DToken(landmarkData) {
    const hasHand = landmarkData && landmarkData.hands && landmarkData.hands.length > 0;
    if (!hasHand) {
        prevHandCenter = null;
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }

    const [hx, hy] = landmarkData.hand_center || [0.5, 0.5];
    const [sx, sy] = landmarkData.shoulder_center || [0.5, 0.35];

    // Hand velocity delta
    let mx = 0.0, my = 0.0;
    if (prevHandCenter !== null) {
        mx = hx - prevHandCenter[0];
        my = hy - prevHandCenter[1];
    }
    prevHandCenter = [hx, hy];

    // Relative to shoulder baseline
    const rx = hx - sx;
    const ry = hy - sy;

    return [hx, hy, mx, my, rx, ry];
}

// ─── Send Token to Backend ───────────────────────────────────────
let lastTokenSentTimestamp = 0;
let lastSentHadHand = false;
const TOKEN_SEND_INTERVAL_MS = 66; // Cap token processing to ~15 FPS to prevent backend flooding

async function sendTokenToBackend(landmarkData, currentFrameId) {
    if (isProcessingToken) {
        droppedFramesCount++;
        return;
    }

    const hasHand = landmarkData && landmarkData.hands && landmarkData.hands.length > 0;
    const now = performance.now();

    // If no hand is present and the backend was already notified, do not continuously send empty POSTs
    if (!hasHand) {
        if (!lastSentHadHand) {
            return; // Backend already reset to NO_HAND; suppress continuous requests
        }
    } else {
        // Enforce 15 FPS cadence (~66ms) when tracking hands to prevent overwhelming the server
        if (now - lastTokenSentTimestamp < TOKEN_SEND_INTERVAL_MS) {
            return;
        }
    }

    lastTokenSentTimestamp = now;
    lastSentHadHand = hasHand;
    isProcessingToken = true;
    const token = compute6DToken(landmarkData);
    const captureTime = now;

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
                timestamp_ms: captureTime,
                pose: landmarkData.pose,
                hand_center: landmarkData.hand_center,
                shoulder_center: landmarkData.shoulder_center,
                has_hand: hasHand
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

    // 6D Token Values
    if (data.token && data.token.length >= 6) {
        tkHx.textContent = data.token[0].toFixed(2);
        tkHy.textContent = data.token[1].toFixed(2);
        tkMx.textContent = data.token[2].toFixed(2);
        tkMy.textContent = data.token[3].toFixed(2);
        tkRx.textContent = data.token[4].toFixed(2);
        tkRy.textContent = data.token[5].toFixed(2);
    }

    // Active Word Extraction for Immediate UI Synchronization
    const activeWordClean = (data.prediction && data.prediction.word && data.prediction.word !== 'BUFFERING' && data.prediction.word !== '--' && (data.prediction.confidence || 0) >= 0.3)
        ? data.prediction.word.replace('_', ' ').toUpperCase()
        : null;

    // Prediction Confidence Bar & Word
    if (data.prediction) {
        currentPrediction = data.prediction;
        const conf = data.prediction.confidence || 0;
        const confPct = Math.round(conf * 100);
        valConfidencePct.textContent = `${confPct}%`;
        confidenceBarFill.style.width = `${confPct}%`;

        // Gate displayed sign: only display sign name if confidence >= 70% or accepted by early decision
        const isConfident = hasHand && (conf >= 0.70 || (data.early_decision && data.early_decision.accepted));
        const rawWord = data.prediction.word;
        const isValidWord = rawWord && rawWord !== 'BUFFERING' && rawWord !== '--';
        const displayWord = isConfident && isValidWord ? rawWord.replace('_', ' ').toUpperCase() : '--';

        valDetectedWord.textContent = displayWord;

        // Update Single Gesture Test Card (Phase 10)
        if (sgValSign) sgValSign.textContent = displayWord;
        if (sgValConf) sgValConf.textContent = `${confPct}%`;
    }

    // Early Decision State Transitions (Phase 11)
    if (data.early_decision) {
        currentEarlyDecision = data.early_decision;
        let determinedState = data.early_decision.state;

        if (!hasHand) {
            determinedState = 'NO_HAND';
        } else if (data.early_decision.accepted || determinedState === 'LOCKED' || determinedState === 'CONFIRMED') {
            determinedState = 'ACCEPTED';
        } else if (data.buffer_status && !data.buffer_status.includes('Ready')) {
            determinedState = 'COLLECTING';
        } else if (data.prediction && data.prediction.confidence > 0.3) {
            determinedState = 'PREDICTING';
        } else {
            determinedState = 'READY';
        }

        updateSystemState(determinedState);
    }

    // Translation Output & Word Buffer Synchronization
    if (data.translation) {
        let text = data.translation.display_text;

        // Remove placeholder text immediately if a valid prediction exists
        if (!text || text.trim() === '') {
            if (activeWordClean) {
                text = activeWordClean.charAt(0) + activeWordClean.slice(1).toLowerCase() + '.';
            } else {
                text = '<em>Waiting for sign gesture input...</em>';
            }
        }

        translatedText.innerHTML = text;
        lastTranslationText = text;

        // Synchronize Word Buffer
        const rawWords = (data.translation.raw_words && data.translation.raw_words.length > 0)
            ? data.translation.raw_words
            : (activeWordClean ? [activeWordClean] : []);
        updateWordBuffer(rawWords);
    }
}

function updateWordBuffer(words) {
    if (!wordBufferContainer) return;
    const label = wordBufferContainer.querySelector('.buffer-label');
    wordBufferContainer.innerHTML = '';
    if (label) wordBufferContainer.appendChild(label);

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
        if (data.translation) {
            translatedText.innerHTML = data.translation.display_text || '<em>Waiting for sign gesture input...</em>';
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
        translatedText.innerHTML = '<em>Waiting for sign gesture input...</em>';
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

