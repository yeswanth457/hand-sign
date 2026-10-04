# RT-STAMP-SLR: 21-Class Real Webcam Recording Plan

## 1. Executive Summary
- **Primary Objective**: Build a balanced, multi-signer, high-fidelity real webcam dataset for all 21 ISL classes.
- **Root Problem Diagnosed**: Training data currently suffers from severe domain shift (studio 854x480 wide shots vs 640x480 laptop webcam), single-signer overfitting (hello, thank_you, welcome recorded by 1 person), and severe class starvation (16 of 21 classes < 20 samples).
- **Target Standard**: Minimum 30 videos per class (preferred 40 videos per class) across at least 3–5 real human signers.

## 2. All 21 Classes Recording Checklist

| Class | Current Videos | Target (Min / Pref) | Additional Required (Min / Pref) | Current Signers | Signers Required |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **hello** | 30 | 30 / 40 | +0 / +10 | 1 | 2 |
| **thank_you** | 30 | 30 / 40 | +0 / +10 | 1 | 2 |
| **welcome** | 35 | 30 / 40 | +0 / +5 | 1 | 2 |
| **goodbye** | 8 | 30 / 40 | +22 / +32 | 4 | 2+ (diversity) |
| **yes** | 17 | 30 / 40 | +13 / +23 | 15 | 2+ (diversity) |
| **no** | 17 | 30 / 40 | +13 / +23 | 17 | 2+ (diversity) |
| **please** | 18 | 30 / 40 | +12 / +22 | 18 | 2+ (diversity) |
| **sorry** | 9 | 30 / 40 | +21 / +31 | 6 | 2+ (diversity) |
| **help** | 15 | 30 / 40 | +15 / +25 | 14 | 2+ (diversity) |
| **stop** | 7 | 30 / 40 | +23 / +33 | 5 | 2+ (diversity) |
| **water** | 19 | 30 / 40 | +11 / +21 | 19 | 2+ (diversity) |
| **food** | 19 | 30 / 40 | +11 / +21 | 18 | 2+ (diversity) |
| **school** | 38 | 30 / 40 | +0 / +2 | 38 | 2+ (diversity) |
| **teacher** | 17 | 30 / 40 | +13 / +23 | 16 | 2+ (diversity) |
| **mother** | 18 | 30 / 40 | +12 / +22 | 18 | 2+ (diversity) |
| **father** | 16 | 30 / 40 | +14 / +24 | 16 | 2+ (diversity) |
| **sister** | 17 | 30 / 40 | +13 / +23 | 15 | 2+ (diversity) |
| **brother** | 7 | 30 / 40 | +23 / +33 | 5 | 2+ (diversity) |
| **friend** | 38 | 30 / 40 | +0 / +2 | 37 | 2+ (diversity) |
| **house** | 3 | 30 / 40 | +27 / +37 | 3 | 2+ (diversity) |
| **work** | 3 | 30 / 40 | +27 / +37 | 1 | 2 |


## 3. Verified ISLRTC / Standard ISL Gesture Protocol Definitions
To ensure semantic validity and eliminate invented gestures, all recordings MUST adhere to official Indian Sign Language definitions:

- **0. hello**: Dominant open hand raised near temple/forehead with palm outward, moving gently outward in a welcoming wave/salute gesture.
- **1. thank_you**: Dominant open flat hand touches chin or lips with fingertips, then extends smoothly forward and outward toward the conversational partner with palm up/forward.
- **2. welcome**: Both open flat hands held out in front of torso, palms facing inward/upward, gently sweeping inward toward the body in an inviting arc.
- **3. goodbye**: Open flat hand raised to shoulder/head level, fingers fluttering or entire palm oscillating side to side in an isolated waving motion.
- **4. yes**: Dominant hand forms a closed fist held at chest level, nodding up and down from the wrist like an affirmative head nod.
- **5. no**: Index and middle fingers extended together, meeting the thumb and pinching closed repeatedly, or index finger waving side to side in a firm negation motion.
- **6. please**: Dominant open flat palm placed flat against the center of chest, rotating in a gentle clockwise circular rubbing motion.
- **7. sorry**: Dominant hand forms an 'A' fist placed against chest/sternum, rotating in a continuous circular rubbing motion with an apologetic facial expression.
- **8. help**: Non-dominant flat palm held horizontal facing upward; dominant hand forms a closed fist with thumb up ('A' shape) resting on palm, moving upward together.
- **9. stop**: Dominant flat hand with fingers together, held vertically with palm facing outward directly toward the camera, or brought down sharply onto horizontal non-dominant palm.
- **10. water**: Dominant hand forms 'W' handshape (index, middle, ring fingers spread upright) or cupped hand brought to lips, tapping chin or lips twice.
- **11. food**: Dominant hand forms a flattened 'O' (fingertips touching thumb), brought to lips repeatedly in a natural eating motion.
- **12. school**: Both flat open palms facing each other horizontally; dominant palm claps down firmly onto non-dominant palm twice.
- **13. teacher**: Both hands form 'O' or pinch shapes at temple level, moving forward and opening, followed by standard agent/person marker (hands parallel moving down torso).
- **14. mother**: Dominant index finger or open thumb taps or brushes cheek/side of chin twice, or 'M' handshape tapped on chin.
- **15. father**: Dominant hand open with thumb tapping center of forehead or temple twice, representing the traditional male marker in ISL.
- **16. sister**: Dominant index finger touches cheek (female marker) followed by both index fingers extended parallel side-by-side touching together.
- **17. brother**: Dominant thumb taps forehead (male marker) followed by both index fingers extended parallel side-by-side touching together.
- **18. friend**: Both hands form bent index fingers (hooks), interlocking together once, then reversing and interlocking again (or right wrist tapping over left wrist).
- **19. house**: Both open flat hands placed diagonally opposite each other, fingertips touching at an apex angle (roof shape) then moving downward vertically (walls).
- **20. work**: Both hands in closed fists with wrists facing downward; dominant fist's heel/wrist taps down firmly on non-dominant fist's wrist twice.

## 4. Controlled Webcam Recording Rules
1. **Fixed Camera Resolution**: Exactly 640x480 at 30 FPS.
2. **Signer Distance**: 0.7 to 1.2 meters from webcam; head and upper torso clearly visible.
3. **Gesture Cadence**: Rest position (hands down/neutral) -> 3-second active gesture execution -> Return to neutral rest position.
4. **Real-Time Quality Gate**: Video is accepted ONLY if MediaPipe detects >= 15 frames of valid hand landmarks; corrupted/no-hand clips are rejected immediately.
