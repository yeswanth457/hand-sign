// ─── MediaPipe Module Placeholders (Dynamically loaded to guarantee instant UI startup) ───
let FilesetResolver = null;
let HandLandmarker = null;
let PoseLandmarker = null;

// ─── Step 9/11: Global Frontend Error Logging ────────────────────
window.addEventListener("error", (event) => {
    console.error("[GLOBAL JS ERROR]", event.error || event.message);
});

window.addEventListener("unhandledrejection", (event) => {
    console.error("[GLOBAL PROMISE ERROR]", event.reason);
});

// Safe numeric formatting to prevent TypeError on null/undefined values
function safeFixed(value, digits = 1) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : "0.0";
}

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

// ─── DOM Element References ──────────────────────────────────────
let videoEl = null;
let canvasEl = null;
let ctx = null;
let motionGraphCanvas = null;
let graphCtx = null;

let btnToggleWebcam = null;
let btnSimulateDemo = null;
let btnSimulateSelected = null;
let btnTrainModel = null;
let btnTTS = null;
let btnClear = null;

let selectVocab = null;
let valMotionEnergy = null;
let decisionStateBadge = null;

// Debug HUD Elements (Phase 3)
let dbgCameraStream = null;
let dbgReadyState = null;
let dbgDimensions = null;
let dbgCurrentTime = null;
let dbgLoopStatus = null;
let dbgFrameId = null;
let dbgDetectorStatus = null;
let dbgHandDetected = null;
let dbgLandmarkCount = null;
let dbgFpsPill = null;

let valMpHandTime = null;
let valMpPoseTime = null;
let valRenderTime = null;
let valTokenCalcTime = null;
let valBackendRtt = null;

// Single Gesture Test Mode Elements (Phase 10)
let sgTargetButtons = [];
let btnClearSgTest = null;
let sgValSign = null;
let sgValConf = null;
let sgValState = null;

// 6D/12D Token Boxes
let tkHx = null;
let tkHy = null;
let tkMx = null;
let tkMy = null;
let tkRx = null;
let tkRy = null;

// Translation Output
let currentLangLabel = null;
let translatedText = null;
let wordBufferContainer = null;
let valDetectedWord = null;
let valConfidencePct = null;
let confidenceBarFill = null;
let valThreshold = null;

function refreshDOMReferences() {
    videoEl = document.getElementById('webcamVideo');
    canvasEl = document.getElementById('landmarkCanvas');
    ctx = canvasEl ? canvasEl.getContext('2d') : null;
    motionGraphCanvas = document.getElementById('motionGraphCanvas');
    graphCtx = motionGraphCanvas ? motionGraphCanvas.getContext('2d') : null;

    btnToggleWebcam = document.getElementById('btnToggleWebcam');
    btnSimulateDemo = document.getElementById('btnSimulateDemo');
    btnSimulateSelected = document.getElementById('btnSimulateSelected');
    btnTrainModel = document.getElementById('btnTrainModel');
    btnTTS = document.getElementById('btnTTS');
    btnClear = document.getElementById('btnClear');

    selectVocab = document.getElementById('selectVocab');
    valMotionEnergy = document.getElementById('valMotionEnergy');
    decisionStateBadge = document.getElementById('decisionStateBadge');

    dbgCameraStream = document.getElementById('dbgCameraStream');
    dbgReadyState = document.getElementById('dbgReadyState');
    dbgDimensions = document.getElementById('dbgDimensions');
    dbgCurrentTime = document.getElementById('dbgCurrentTime');
    dbgLoopStatus = document.getElementById('dbgLoopStatus');
    dbgFrameId = document.getElementById('dbgFrameId');
    dbgDetectorStatus = document.getElementById('dbgDetectorStatus');
    dbgHandDetected = document.getElementById('dbgHandDetected');
    dbgLandmarkCount = document.getElementById('dbgLandmarkCount');
    dbgFpsPill = document.getElementById('dbgFpsPill');

    valMpHandTime = document.getElementById('valMpHandTime');
    valMpPoseTime = document.getElementById('valMpPoseTime');
    valRenderTime = document.getElementById('valRenderTime');
    valTokenCalcTime = document.getElementById('valTokenCalcTime');
    valBackendRtt = document.getElementById('valBackendRtt');

    sgTargetButtons = Array.from(document.querySelectorAll('.sg-target-btn'));
    btnClearSgTest = document.getElementById('btnClearSgTest');
    sgValSign = document.getElementById('sgValSign');
    sgValConf = document.getElementById('sgValConf');
    sgValState = document.getElementById('sgValState');

    tkHx = document.getElementById('tkHx');
    tkHy = document.getElementById('tkHy');
    tkMx = document.getElementById('tkMx');
    tkMy = document.getElementById('tkMy');
    tkRx = document.getElementById('tkRx');
    tkRy = document.getElementById('tkRy');

    currentLangLabel = document.getElementById('currentLangLabel');
    translatedText = document.getElementById('translatedText');
    wordBufferContainer = document.getElementById('wordBufferContainer');
    valDetectedWord = document.getElementById('valDetectedWord');
    valConfidencePct = document.getElementById('valConfidencePct');
    confidenceBarFill = document.getElementById('confidenceBarFill');
    valThreshold = document.getElementById('valThreshold');
}

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

// ─── Robust Event Listeners (Bound First) ────────────────────────
function initEventListeners() {
    console.log("[UI] Binding all UI event listeners...");

    // Start / Stop Webcam Button
    const btnToggle = document.getElementById('btnToggleWebcam');
    if (btnToggle) {
        btnToggle.addEventListener('click', async () => {
            console.log("[UI] Start Webcam clicked");
            await toggleWebcam();
        });
        console.log("[UI] Start Webcam button listener attached.");
    } else {
        console.error("[UI] Start Webcam button (#btnToggleWebcam) not found in DOM");
    }

    // Single Gesture Test buttons
    const targetBtns = document.querySelectorAll('.sg-target-btn');
    targetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.target;
            if (!target) return;
            singleTestTargetSign = target;
            targetBtns.forEach(b => {
                if (b.dataset.target) b.classList.toggle('active', b.dataset.target === target);
            });
            console.log(`[UI] Gesture button clicked: ${target}`);
            console.log(`[Single Gesture Mode] Target set to: ${target.toUpperCase()}`);
        });
    });
    console.log(`[UI] Attached ${targetBtns.length} gesture test button listeners.`);

    // Clear Single Gesture Test button
    const btnClearTest = document.getElementById('btnClearSgTest');
    if (btnClearTest) {
        btnClearTest.addEventListener('click', () => {
            console.log("[UI] Clear Single Gesture Test clicked");
            resetSingleGestureTest();
        });
    }

    // English / Tamil Language Buttons
    const btnEng = document.getElementById('btnLangEng');
    if (btnEng) {
        btnEng.addEventListener('click', () => {
            console.log("[UI] Language button clicked: english");
            setLanguage('english');
        });
    }

    const btnTam = document.getElementById('btnLangTam');
    if (btnTam) {
        btnTam.addEventListener('click', () => {
            console.log("[UI] Language button clicked: tamil");
            setLanguage('tamil');
        });
    }

    // Simulation / Training / Audio / Buffer Clear Buttons
    const btnSimDemo = document.getElementById('btnSimulateDemo');
    if (btnSimDemo) btnSimDemo.addEventListener('click', () => simulateSign('water'));

    const btnSimSel = document.getElementById('btnSimulateSelected');
    if (btnSimSel) btnSimSel.addEventListener('click', () => {
        const sel = document.getElementById('selectVocab');
        const word = sel ? sel.value : null;
        if (word) simulateSign(word);
    });

    const btnTrain = document.getElementById('btnTrainModel');
    if (btnTrain) btnTrain.addEventListener('click', trainModel);

    const btnSpeak = document.getElementById('btnTTS');
    if (btnSpeak) btnSpeak.addEventListener('click', speakTranslation);

    const btnClr = document.getElementById('btnClear');
    if (btnClr) btnClr.addEventListener('click', clearSentence);
}

// Global exports for inline HTML or external access
window.setLanguage = setLanguage;
window.simulateSign = simulateSign;
window.trainModel = trainModel;
window.speakTranslation = speakTranslation;
window.clearSentence = clearSentence;
window.resetSingleGestureTest = resetSingleGestureTest;
window.toggleWebcam = toggleWebcam;
window.startWebcam = startWebcam;
window.stopWebcam = stopWebcam;
window.loadDatasetBrowser = loadDatasetBrowser;

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

// ─── Robust Initialization Pipeline (Step 4) ────────────────────
function initializeApp() {
    console.log("[UI] Initializing RT-STAMP-SLR frontend application...");
    
    // 1. Refresh all DOM node references
    refreshDOMReferences();

    // 2. Attach all event listeners FIRST so buttons work immediately
    initEventListeners();
    initDatasetBrowserListeners();

    // 3. Set clean initial UI state
    try {
        setCameraState('CAMERA_OFF', 'Application initialized, camera idle.');
        setPipelineState('PIPELINE_STARTING', 'Initializing MediaPipe...');
        updateSystemState('CAMERA_OFF', 'Application initialized, camera idle.');

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
    } catch (e) {
        console.warn("[UI] Non-blocking initial UI setup notice:", e);
    }

    // 4. Load vocabulary and draw initial graph
    try {
        loadVocabulary();
        drawMotionGraph();
    } catch (e) {
        console.warn("[UI] Non-blocking vocab/graph notice:", e);
    }

    // 5. Initialize MediaPipe in background (non-blocking)
    initMediaPipe().catch(e => console.warn("[MediaPipe] Background init notice:", e));

    // 6. Auto-load dataset explorer page 1
    try {
        loadDatasetBrowser(0, DATASET_PAGE_SIZE);
    } catch (e) {
        console.warn("[UI] Non-blocking dataset explorer notice:", e);
    }

    // 7. Reset server sentence buffer
    fetch('/api/clear_sentence', { method: 'POST' }).catch(() => {});

    console.log("[UI] Frontend initialization completed successfully.");
}

// Execute immediately if DOM is already parsed, or wait for DOMContentLoaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
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
        if (!FilesetResolver || !HandLandmarker || !PoseLandmarker) {
            let mpVision = null;
            try {
                mpVision = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs");
            } catch (e1) {
                console.warn('[MediaPipe] Direct ESM import failed, checking window:', e1);
                mpVision = window;
            }
            FilesetResolver = mpVision?.FilesetResolver || window.FilesetResolver;
            HandLandmarker = mpVision?.HandLandmarker || window.HandLandmarker;
            PoseLandmarker = mpVision?.PoseLandmarker || window.PoseLandmarker;
        }

        if (!FilesetResolver || !HandLandmarker || !PoseLandmarker) {
            throw new Error('MediaPipe Tasks Vision exports could not be loaded');
        }

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
    if (!videoEl || !btnToggleWebcam || !canvasEl) {
        refreshDOMReferences();
    }

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

                const gestureWord = (hasHand && currentPrediction && currentPrediction.word && currentPrediction.word !== '--')
                    ? currentPrediction.word.replace('_', ' ').toUpperCase()
                    : '--';
                const gestureConf = (currentPrediction && currentPrediction.confidence !== null && currentPrediction.confidence !== undefined)
                    ? `${safeFixed(Number(currentPrediction.confidence) * 100, 1)}%`
                    : '0.0%';
                const gestureState = (currentEarlyDecision && currentEarlyDecision.state) || (hasHand ? 'COLLECTING' : 'NO_HAND');

                console.log(
                    `\n[RT-STAMP-SLR DEBUG]\n` +
                    `Frame: ${frameId}\n` +
                    `FPS: ${webcamFps}\n` +
                    `\n` +
                    `Camera: ${streamActive ? 'STREAM ACTIVE' : 'STREAM INACTIVE'}\n` +
                    `Video: ${videoEl.videoWidth}x${videoEl.videoHeight} readyState=${videoEl.readyState}\n` +
                    `Video currentTime: ${safeFixed(videoEl ? videoEl.currentTime : 0, 2)}s\n` +
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
                    `Detection confidence: ${safeFixed(rawDetectionConf, 4)}\n` +
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
                    `  Hand MP: ${safeFixed(mpHandInferenceMs, 1)}ms\n` +
                    `  Render: ${safeFixed(renderDurationMs, 1)}ms\n` +
                    `  Token calc: ${safeFixed(tokenCalculationMs, 1)}ms\n` +
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
                    const w0Z = (w0 && w0.length >= 3 && w0[2] !== undefined && w0[2] !== null) ? `, ${safeFixed(w0[2], 4)}` : '';
                    const i8Z = (i8 && i8.length >= 3 && i8[2] !== undefined && i8[2] !== null) ? `, ${safeFixed(i8[2], 4)}` : '';
                    console.log(
                        `[LANDMARK COORD VERIFICATION]\n` +
                        `WRIST #0: MP (${safeFixed(w0 ? w0[0] : 0, 4)}, ${safeFixed(w0 ? w0[1] : 0, 4)}${w0Z}) | Video: ${vw}x${vh} | Canvas: ${cw}x${ch} | Draw: (${safeFixed((w0 ? w0[0] : 0) * cw, 1)}, ${safeFixed((w0 ? w0[1] : 0) * ch, 1)})\n` +
                        `INDEX TIP #8: MP (${safeFixed(i8 ? i8[0] : 0, 4)}, ${safeFixed(i8 ? i8[1] : 0, 4)}${i8Z}) | Video: ${vw}x${vh} | Canvas: ${cw}x${ch} | Draw: (${safeFixed((i8 ? i8[0] : 0) * cw, 1)}, ${safeFixed((i8 ? i8[1] : 0) * ch, 1)})`
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
        dbgCurrentTime.textContent = videoEl && Number.isFinite(videoEl.currentTime) ? `${safeFixed(videoEl.currentTime, 2)}s` : '0.00s';
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

    if (valMpHandTime) valMpHandTime.textContent = `${safeFixed(mpHandInferenceMs, 1)}ms`;
    if (valMpPoseTime) valMpPoseTime.textContent = `${safeFixed(mpPoseInferenceMs, 1)}ms`;
    if (valRenderTime) valRenderTime.textContent = `${safeFixed(renderDurationMs, 1)}ms`;
    if (valTokenCalcTime) valTokenCalcTime.textContent = `${safeFixed(tokenCalculationMs, 1)}ms`;
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


// Adaptive dual-hand landmark smoothing & tracking state
let activeGestureHand = null;         // "Right", "Left", "Both", or null when unlocked
let lockedHandCenter = null;          // [x, y]
let lockedHandMissingFrames = 0;      // Streak of missing frames for locked hand
const MAX_LOCKED_HAND_GRACE_FRAMES = 4; // Grace frames before resetting gesture

let prevLeftCenter = null;
let prevRightCenter = null;
let currentSequenceId = Date.now();
let consecutiveNoHandFrames = 0;
const CONSECUTIVE_NO_HAND_FOR_RESET = 4;

function resetTokenizerState() {
    prevLeftCenter = null;
    prevRightCenter = null;
    activeGestureHand = null;
    lockedHandCenter = null;
    lockedHandMissingFrames = 0;
    currentSequenceId = Date.now();
    consecutiveNoHandFrames = 0;
}

// ─── Build Landmark Data (Two-Hand Landmark Extraction & Separation) ───
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
        raw_hands: 0,
        left_detected: false,
        right_detected: false,
        left_confidence: 0.0,
        right_confidence: 0.0,
        active_hands: "NONE",
        selected_hand: "Right",
        selected_hand_index: 0,
        selected_hand_confidence: 1.0,
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

    const midX = data.shoulder_center ? data.shoulder_center[0] : 0.5;

    // Stable Hand Tracking State supporting BOTH hands (Steps 3, 4, 8)
    const detectedHandsList = [];
    const detectedHandMeta = [];
    if (handResults && handResults.landmarks && handResults.landmarks.length > 0) {
        handResults.landmarks.forEach((hand, idx) => {
            if (hand && hand.length === 21) {
                const pts = hand.map(lm => [lm.x, lm.y, lm.z]);
                detectedHandsList.push(pts);
                let label = 'Right';
                let score = 1.0;
                if (handResults.handedness && handResults.handedness[idx] && handResults.handedness[idx][0]) {
                    label = handResults.handedness[idx][0].categoryName || ('Hand_' + idx);
                    score = handResults.handedness[idx][0].score || 1.0;
                }
                detectedHandMeta.push({ label, score });
            }
        });
    }

    const rawHandCount = detectedHandsList.length;
    data.raw_hands = rawHandCount;

    if (rawHandCount === 0) {
        // No hands detected in frame
        if (activeGestureHand !== null) {
            lockedHandMissingFrames++;
            if (lockedHandMissingFrames <= MAX_LOCKED_HAND_GRACE_FRAMES) {
                data.selected_hand = activeGestureHand;
            } else {
                resetTokenizerState();
                data.selected_hand = "NONE";
            }
        } else {
            data.selected_hand = "NONE";
        }
        return data;
    }

    // At least one hand detected in frame
    const centers = detectedHandsList.map(h => calculateHandCenter(h));

    // Step 3: Log [TWO HAND DEBUG] when 2 hands are detected
    if (rawHandCount >= 2 && detectedHandMeta && detectedHandMeta.length >= 2 && centers && centers.length >= 2) {
        console.log(
            `[TWO HAND DEBUG]\n` +
            `raw_hand_count=2\n` +
            `hand_0_label = ${detectedHandMeta[0].label || 'UNKNOWN'}\n` +
            `hand_0_confidence = ${safeFixed(detectedHandMeta[0].score, 4)}\n` +
            `hand_0_center = [${safeFixed(centers[0] ? centers[0][0] : 0, 4)}, ${safeFixed(centers[0] ? centers[0][1] : 0, 4)}]\n` +
            `hand_1_label = ${detectedHandMeta[1].label || 'UNKNOWN'}\n` +
            `hand_1_confidence = ${safeFixed(detectedHandMeta[1].score, 4)}\n` +
            `hand_1_center = [${safeFixed(centers[1] ? centers[1][0] : 0, 4)}, ${safeFixed(centers[1] ? centers[1][1] : 0, 4)}]`
        );
    }

    // Classify each detected hand into Left or Right deterministically
    detectedHandsList.forEach((handPts, idx) => {
        const meta = detectedHandMeta[idx];
        const center = centers[idx];
        if (meta.label === 'Left' && !data.left_hand) {
            data.left_hand = handPts;
            data.left_hand_center = center;
            data.left_detected = true;
            data.left_confidence = meta.score;
        } else if (meta.label === 'Right' && !data.right_hand) {
            data.right_hand = handPts;
            data.right_hand_center = center;
            data.right_detected = true;
            data.right_confidence = meta.score;
        } else if (!data.left_hand && center[0] < midX) {
            data.left_hand = handPts;
            data.left_hand_center = center;
            data.left_detected = true;
            data.left_confidence = meta.score;
        } else if (!data.right_hand) {
            data.right_hand = handPts;
            data.right_hand_center = center;
            data.right_detected = true;
            data.right_confidence = meta.score;
        }
    });

    // Spatial fallback if both hands are detected but assigned to the same side
    if (rawHandCount >= 2 && (!data.left_hand || !data.right_hand)) {
        if (centers[0][0] < centers[1][0]) {
            data.left_hand = detectedHandsList[0];
            data.left_hand_center = centers[0];
            data.left_detected = true;
            data.left_confidence = detectedHandMeta[0].score;
            data.right_hand = detectedHandsList[1];
            data.right_hand_center = centers[1];
            data.right_detected = true;
            data.right_confidence = detectedHandMeta[1].score;
        } else {
            data.left_hand = detectedHandsList[1];
            data.left_hand_center = centers[1];
            data.left_detected = true;
            data.left_confidence = detectedHandMeta[1].score;
            data.right_hand = detectedHandsList[0];
            data.right_hand_center = centers[0];
            data.right_detected = true;
            data.right_confidence = detectedHandMeta[0].score;
        }
    }

    // Populate hands array for canvas rendering and active_hands
    if (data.left_detected && data.right_detected) {
        data.active_hands = "Left+Right";
        activeGestureHand = "Both";
        data.hands = [data.left_hand, data.right_hand];
        data.hand_count = 2;
        data.hand_center = data.right_hand_center || data.left_hand_center;
    } else if (data.left_detected) {
        data.active_hands = "Left";
        activeGestureHand = "Left";
        data.hands = [data.left_hand];
        data.hand_count = 1;
        data.hand_center = data.left_hand_center;
    } else {
        data.active_hands = "Right";
        activeGestureHand = "Right";
        data.hands = [data.right_hand];
        data.hand_count = 1;
        data.hand_center = data.right_hand_center;
    }

    data.selected_hand = activeGestureHand;
    lockedHandMissingFrames = 0;
    return data;
}

// ─── 12D Token Computation (Browser-Side) ─────────────────────────
function compute12DToken(landmarkData) {
    const hasHand = landmarkData && (landmarkData.left_detected || landmarkData.right_detected || (landmarkData.hands && landmarkData.hands.length > 0));
    if (!hasHand) {
        consecutiveNoHandFrames++;
        if (consecutiveNoHandFrames >= CONSECUTIVE_NO_HAND_FOR_RESET) {
            resetTokenizerState();
        }
        return null;
    }

    consecutiveNoHandFrames = 0;

    const ls = (landmarkData.pose && landmarkData.pose.LS) ? landmarkData.pose.LS : [0.4, 0.35];
    const rs = (landmarkData.pose && landmarkData.pose.RS) ? landmarkData.pose.RS : [0.6, 0.35];

    let leftToken = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    if (landmarkData.left_hand_center) {
        const lhx = landmarkData.left_hand_center[0];
        const lhy = landmarkData.left_hand_center[1];
        let lmx = 0.0, lmy = 0.0;
        if (prevLeftCenter !== null) {
            const d = Math.hypot(lhx - prevLeftCenter[0], lhy - prevLeftCenter[1]);
            if (d <= 0.3) {
                lmx = lhx - prevLeftCenter[0];
                lmy = lhy - prevLeftCenter[1];
            }
        }
        prevLeftCenter = [lhx, lhy];
        const lrx = lhx - ls[0];
        const lry = lhy - ls[1];
        leftToken = [lhx, lhy, lmx, lmy, lrx, lry];
    } else {
        prevLeftCenter = null;
    }

    let rightToken = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    if (landmarkData.right_hand_center) {
        const rhx = landmarkData.right_hand_center[0];
        const rhy = landmarkData.right_hand_center[1];
        let rmx = 0.0, rmy = 0.0;
        if (prevRightCenter !== null) {
            const d = Math.hypot(rhx - prevRightCenter[0], rhy - prevRightCenter[1]);
            if (d <= 0.3) {
                rmx = rhx - prevRightCenter[0];
                rmy = rhy - prevRightCenter[1];
            }
        }
        prevRightCenter = [rhx, rhy];
        const rrx = rhx - rs[0];
        const rry = rhy - rs[1];
        rightToken = [rhx, rhy, rmx, rmy, rrx, rry];
    } else {
        prevRightCenter = null;
    }

    const token = [...leftToken, ...rightToken];
    updateTokenDisplay(token, true);
    return token;
}

function updateTokenDisplay(token, hasHand) {
    if (!hasHand || !token || token.length < 12) {
        if (tkHx) tkHx.textContent = '--';
        if (tkHy) tkHy.textContent = '--';
        if (tkMx) tkMx.textContent = '--';
        if (tkMy) tkMy.textContent = '--';
        if (tkRx) tkRx.textContent = '--';
        if (tkRy) tkRy.textContent = '--';
        return;
    }
    // Display dominant/active hand coordinates in UI HUD
    const offset = (token[6] !== 0 || token[7] !== 0) ? 6 : 0;
    if (tkHx) tkHx.textContent = safeFixed(token[offset + 0], 2);
    if (tkHy) tkHy.textContent = safeFixed(token[offset + 1], 2);
    if (tkMx) tkMx.textContent = safeFixed(token[offset + 2], 2);
    if (tkMy) tkMy.textContent = safeFixed(token[offset + 3], 2);
    if (tkRx) tkRx.textContent = safeFixed(token[offset + 4], 2);
    if (tkRy) tkRy.textContent = safeFixed(token[offset + 5], 2);
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

    const hasHand = landmarkData && (landmarkData.left_detected || landmarkData.right_detected || (landmarkData.hands && landmarkData.hands.length > 0));
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
    const token = compute12DToken(landmarkData);
    const captureTime = now;

    if (now - _lastWebFrameLog >= 1000) {
        _lastWebFrameLog = now;
        console.log(`[WEB FRAME] Sending frame to /api/process_token (frame_id=${currentFrameId}, seq_id=${currentSequenceId}, hasHand=${hasHand}, hands=${landmarkData.active_hands || "NONE"})`);
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
                hands: landmarkData.hands,
                hand_count: landmarkData.hand_count || (hasHand ? 1 : 0),
                raw_hands: landmarkData.raw_hands || 0,
                left_detected: landmarkData.left_detected || false,
                right_detected: landmarkData.right_detected || false,
                left_confidence: landmarkData.left_confidence || 0.0,
                right_confidence: landmarkData.right_confidence || 0.0,
                active_hands: landmarkData.active_hands || "NONE",
                primary_hand: landmarkData.selected_hand || "Right",
                selected_hand: landmarkData.selected_hand || "Right",
                selected_hand_index: landmarkData.selected_hand_index !== undefined ? landmarkData.selected_hand_index : 0,
                selected_hand_confidence: landmarkData.selected_hand_confidence || 1.0,
                camera_fps: webcamFps || 30.0,
                token_fps: Math.round(1000 / TOKEN_SEND_INTERVAL_MS)
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
    if (!data) return;

    // ── [GESTURE DEBUG] Mirror backend diagnostic block in browser console ──
    // Only log when a full 25-token prediction is available (buffer_status === "25/25")
    if (data.buffer_status === '25/25') {
        const _now = performance.now();
        if (_now - _lastNoWebLog >= 500) {
            _lastNoWebLog = _now;
            console.log(
                `\n[GESTURE DEBUG]\n` +
                `21CLASS        = ${data.primary_class || '--'}\n` +
                `21CLASS_CONF   = ${safeFixed(data.primary_confidence, 4)}\n` +
                `NO_PROBABILITY = ${safeFixed(data.no_probability !== undefined && data.no_probability !== null ? data.no_probability : (data.binary_no ? data.binary_no.no_probability : 0), 4)}\n` +
                `NO_CONFIRMATIONS = ${data.no_confirmations !== undefined && data.no_confirmations !== null ? data.no_confirmations : (data.binary_no ? data.binary_no.consecutive_count : 0)}\n` +
                `NO_CONFIRMED   = ${data.no_confirmed !== undefined && data.no_confirmed !== null ? data.no_confirmed : (data.binary_no ? data.binary_no.no_confirmed : false)}\n` +
                `FINAL_CLASS    = ${data.final_class || '--'}`
            );
            // Also log token coordinates for NO-token coordinate verification
            if (data.token && data.token.length >= 6) {
                console.log(
                    `[NO TOKEN]\n` +
                    `Hx=${safeFixed(data.token[0], 4)}\n` +
                    `Hy=${safeFixed(data.token[1], 4)}\n` +
                    `Mx=${safeFixed(data.token[2], 4)}\n` +
                    `My=${safeFixed(data.token[3], 4)}\n` +
                    `Rx=${safeFixed(data.token[4], 4)}\n` +
                    `Ry=${safeFixed(data.token[5], 4)}`
                );
            }
        }
    } else if (data.binary_no) {
        // Still log binary_no for partial buffers at lower rate
        const _now = performance.now();
        if (_now - _lastNoWebLog >= 1000) {
            _lastNoWebLog = _now;
            console.log(
                `[NO WEB] prob=${safeFixed(data.binary_no.no_probability, 4)} | ` +
                `pred=${data.binary_no.no_prediction || '--'} | ` +
                `confirmed=${data.binary_no.no_confirmed || false} | ` +
                `count=${data.binary_no.consecutive_count || 0}`
            );
        }
    }

    // Motion Energy & Threshold
    if (valMotionEnergy) {
        valMotionEnergy.textContent = (data.motion_energy !== undefined && data.motion_energy !== null)
            ? safeFixed(data.motion_energy, 4)
            : '0.0000';
    }
    if (valThreshold) {
        valThreshold.textContent = (data.threshold !== undefined && data.threshold !== null)
            ? safeFixed(data.threshold, 4)
            : '0.0150';
    }
    if (data.threshold !== undefined && data.threshold !== null) {
        motionHistory.shift();
        motionHistory.push(Number(data.motion_energy) || 0);
        thresholdHistory.shift();
        thresholdHistory.push(Number(data.threshold) || 0);
        drawMotionGraph();
    }

    // 12D Token Values (Real values from backend / tokenizer)
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
            ? (data.no_probability !== undefined && data.no_probability !== null ? data.no_probability : (data.binary_no ? data.binary_no.no_probability : 0.90))
            : ((data.primary_confidence !== undefined && data.primary_confidence !== null) ? data.primary_confidence : ((data.prediction && data.prediction.confidence) || 0));
        const confPct = Math.round((Number(conf) || 0) * 100);
        currentPrediction = { word: rawWord, confidence: Number(conf) || 0 };

        let displayWord = '--';
        if (!hasHand) {
            displayWord = '--';
            if (valConfidencePct) valConfidencePct.textContent = '0%';
            if (confidenceBarFill) confidenceBarFill.style.width = '0%';
        } else if (rawWord === 'COLLECTING GESTURE...') {
            displayWord = 'COLLECTING GESTURE...';
            if (valConfidencePct) valConfidencePct.textContent = '0%';
            if (confidenceBarFill) confidenceBarFill.style.width = '0%';
        } else if (rawWord && rawWord !== '--' && rawWord !== 'BUFFERING' && rawWord !== 'WAITING FOR CLEAR GESTURE') {
            displayWord = rawWord.replace('_', ' ').toUpperCase();
            if (valConfidencePct) valConfidencePct.textContent = `${confPct}%`;
            if (confidenceBarFill) confidenceBarFill.style.width = `${confPct}%`;
        } else {
            displayWord = '--';
            if (valConfidencePct) valConfidencePct.textContent = `${confPct}%`;
            if (confidenceBarFill) confidenceBarFill.style.width = `${confPct}%`;
        }

        if (valDetectedWord) valDetectedWord.textContent = displayWord;

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
                    else if (activeSign === 'thalapathy') text = 'தளபதி (Thalapathy)';
                    else text = (data.translation && data.translation.tamil) ? data.translation.tamil : activeSign;
                } else {
                    if (activeSign === 'water') text = 'Water (தண்ணீர்)';
                    else if (activeSign === 'no') text = 'No (இல்லை)';
                    else if (activeSign === 'please') text = 'Please.';
                    else if (activeSign === 'school') text = 'School';
                    else if (activeSign === 'thalapathy') text = 'Thalapathy';
                    else text = activeSign.charAt(0).toUpperCase() + activeSign.slice(1).replace('_', ' ') + '.';
                }
            } else {
                text = '<em>WAITING FOR HAND GESTURE</em>';
            }
        }

        if (text === 'School.') text = 'School';
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
// Proper function declaration so it is hoisted and available for window.setLanguage = setLanguage at module top level
async function setLanguage(lang) {
    currentLanguage = lang;
    const btnE = document.getElementById('btnLangEng');
    const btnT = document.getElementById('btnLangTam');
    if (btnE) btnE.classList.toggle('active', lang === 'english');
    if (btnT) btnT.classList.toggle('active', lang === 'tamil');
    if (currentLangLabel) currentLangLabel.textContent = lang === 'english' ? 'ENGLISH SENTENCE' : 'TAMIL TRANSLATION (தமிழ்)';

    try {
        const res = await fetch('/api/toggle_language', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: lang })
        });
        const data = await res.json();
        if (data.translation && data.translation.display_text && data.translation.display_text.trim() !== '') {
            if (translatedText) translatedText.innerHTML = data.translation.display_text;
            lastTranslationText = data.translation.display_text;
        } else {
            if (translatedText) translatedText.innerHTML = isWebcamRunning ? '<em>WAITING FOR HAND GESTURE</em>' : '<em>WAITING FOR WEBCAM</em>';
        }
    } catch (e) {
        console.error('Toggle language error:', e);
    }
}

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
        alert(`Model trained successfully!\nBest Accuracy: ${safeFixed((data.training_result ? data.training_result.best_val_acc : 0) * 100, 1)}%`);
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
        const durationStr = row.duration ? `${safeFixed(row.duration, 1)}s` : '--';
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
                <i class="fa-solid fa-star"></i> Quality: ${safeFixed(qs * 100, 0)}% ${qualityLabel}
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

// ─── Dataset Explorer Event Listeners ───────────────────────────
function initDatasetBrowserListeners() {
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
}

