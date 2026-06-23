import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ---- Safety settings ----
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# ---- Screen dimensions ----
SCREEN_W, SCREEN_H = pyautogui.size()

# ---- MediaPipe setup ----
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---- Webcam ----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---- State tracking ----
prev_x, prev_y = 0, 0
smoothing = 5
click_cooldown = 0
scroll_cooldown = 0
action_cooldown = 0
both_eyes_closed_start = None
EYES_CLOSE_DURATION = 2.0  # hold both eyes closed for 2 seconds to quit

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_finger_state(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def is_pinch(hand_landmarks, threshold=0.05):
    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]
    return get_distance(index_tip, thumb_tip) < threshold

def is_fist(fingers):
    return sum(fingers) == 0

def is_thumbs_up(hand_landmarks, fingers):
    """
    Thumbs up: only thumb is up,
    all other fingers folded down,
    thumb pointing upward (y position higher than base)
    """
    if fingers[0] != 1:
        return False
    if fingers[1] != 0 or fingers[2] != 0 or fingers[3] != 0 or fingers[4] != 0:
        return False
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    return thumb_tip.y < thumb_base.y

def is_scroll_up(hand_landmarks, fingers):
    if fingers[1] != 1 or fingers[2] != 1:
        return False
    if fingers[3] != 0 or fingers[4] != 0:
        return False
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    return get_distance(index_tip, middle_tip) < 0.06

def is_scroll_down(hand_landmarks, fingers):
    if fingers[3] != 1 or fingers[4] != 1:
        return False
    if fingers[1] != 0 or fingers[2] != 0:
        return False
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    return get_distance(ring_tip, pinky_tip) < 0.06

def get_eye_ratio(face_landmarks, eye_indices):
    top = face_landmarks.landmark[eye_indices[0]]
    bottom = face_landmarks.landmark[eye_indices[1]]
    left = face_landmarks.landmark[eye_indices[2]]
    right = face_landmarks.landmark[eye_indices[3]]
    vertical = get_distance(top, bottom)
    horizontal = get_distance(left, right)
    if horizontal == 0:
        return 0
    return vertical / horizontal

# Eye landmark indices
LEFT_EYE = [159, 145, 33, 133]
RIGHT_EYE = [386, 374, 362, 263]

print("✅ Gesture Mouse Control Started!")
print("\nControls:")
print("  🖱️  Move hand              → move cursor")
print("  👌 Pinch (index+thumb)    → left click")
print("  ✊ Fist                   → right click")
print("  ☝️✌️  Index+Middle together  → scroll UP")
print("  💍🤙 Ring+Pinky together   → scroll DOWN")
print("  👍 Thumbs up              → open new Chrome tab")
print("  😑 Close BOTH eyes (2s)   → quit app")
print("  Press Q to quit\n")

running = True

while cap.isOpened() and running:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    current_time = time.time()

    # ---- Face gesture: both eyes closed = quit ----
    if face_results.multi_face_landmarks:
        face_lm = face_results.multi_face_landmarks[0]

        left_ratio = get_eye_ratio(face_lm, LEFT_EYE)
        right_ratio = get_eye_ratio(face_lm, RIGHT_EYE)

        EYE_CLOSED_THRESHOLD = 0.15

        both_closed = left_ratio < EYE_CLOSED_THRESHOLD and right_ratio < EYE_CLOSED_THRESHOLD

        if both_closed:
            if both_eyes_closed_start is None:
                both_eyes_closed_start = current_time
            else:
                elapsed = current_time - both_eyes_closed_start
                remaining = EYES_CLOSE_DURATION - elapsed
                cv2.putText(frame, f"😑 Keep eyes closed to quit: {remaining:.1f}s",
                           (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2)
                if elapsed >= EYES_CLOSE_DURATION:
                    print("\n😑 Both eyes closed — quitting!")
                    running = False
        else:
            both_eyes_closed_start = None

        # Show eye ratios for debug
        cv2.putText(frame, f"L:{left_ratio:.2f} R:{right_ratio:.2f}",
                   (w - 150, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # ---- Hand gesture control ----
    if hand_results.multi_hand_landmarks:
        hand_lm = hand_results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        index_tip = hand_lm.landmark[8]
        fingers = get_finger_state(hand_lm)

        # ---- Cursor movement ----
        target_x = int(np.interp(index_tip.x, [0.1, 0.9], [0, SCREEN_W]))
        target_y = int(np.interp(index_tip.y, [0.1, 0.9], [0, SCREEN_H]))
        curr_x = prev_x + (target_x - prev_x) / smoothing
        curr_y = prev_y + (target_y - prev_y) / smoothing
        prev_x, prev_y = curr_x, curr_y
        pyautogui.moveTo(curr_x, curr_y)

        # ---- Thumbs up → new Chrome tab ----
        if is_thumbs_up(hand_lm, fingers) and (current_time - action_cooldown) > 1.5:
            pyautogui.hotkey('ctrl', 't')
            action_cooldown = current_time
            cv2.putText(frame, "👍 NEW TAB OPENED!", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            print("👍 New tab opened!", end="\r")

        # ---- Scroll UP: Index + Middle together ----
        elif is_scroll_up(hand_lm, fingers) and (current_time - scroll_cooldown) > 0.3:
            pyautogui.click()
            time.sleep(0.05)
            pyautogui.press(['up', 'up', 'up', 'up', 'up'])
            scroll_cooldown = current_time
            cv2.putText(frame, "SCROLL UP ⬆️", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # ---- Scroll DOWN: Ring + Pinky together ----
        elif is_scroll_down(hand_lm, fingers) and (current_time - scroll_cooldown) > 0.3:
            pyautogui.click()
            time.sleep(0.05)
            pyautogui.press(['down', 'down', 'down', 'down', 'down'])
            scroll_cooldown = current_time
            cv2.putText(frame, "SCROLL DOWN ⬇️", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)

        # ---- Pinch → left click ----
        elif is_pinch(hand_lm) and (current_time - click_cooldown) > 0.8:
            pyautogui.click()
            click_cooldown = current_time
            cv2.putText(frame, "LEFT CLICK", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ---- Fist → right click ----
        elif is_fist(fingers) and (current_time - click_cooldown) > 0.8:
            pyautogui.rightClick()
            click_cooldown = current_time
            cv2.putText(frame, "RIGHT CLICK", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ---- Finger state display ----
        finger_names = ["T", "I", "M", "R", "P"]
        finger_display = " ".join([
            f"{finger_names[i]}:{'UP' if fingers[i] else 'dn'}"
            for i in range(5)
        ])
        cv2.putText(frame, finger_display, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # ---- Display info ----
    cv2.putText(frame, "Gesture Mouse Control", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "Press Q to quit", (w - 200, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Gesture Mouse Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nStopped ✅")