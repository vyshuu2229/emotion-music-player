import cv2
import mediapipe as mp
import vlc
import threading
from tkinter import Tk, Label
import yt_dlp
import random
import time

# --- Config: change queries if you want different search terms ---
youtube_queries = {
    "happy": ["happy kannada song", "happy hindi song", "feel good kannada song"],
    "sad": ["sad hindi song", "sad kannada song", "emotional kannada song"],
    "neutral": ["calm kannada song", "relax hindi song", "lofi kannada music"]
}

# --- Globals ---
current_mood = ""
current_title = ""
player = None
player_instance = None
stop_event = threading.Event()
player_lock = threading.Lock()

# --- GUI setup ---
root = Tk()
root.title("Emotion Player — AI YouTube Music")
root.geometry("600x120")
mood_label = Label(root, text="Mood: Detecting...", font=("Arial", 18))
mood_label.pack(pady=(12, 2))
song_label = Label(root, text="Song: —", font=("Arial", 12))
song_label.pack(pady=(0, 8))

# --- MediaPipe Face Mesh setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

# --- yt-dlp helper: returns (audio_stream_url, title) ---
def get_youtube_audio_and_title(query):
    ydl_opts = {
        "format": "bestaudio",
        "quiet": True,
        "noplaylist": True,
        "default_search": "ytsearch1"
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=False)
        if "entries" in info:
            info = info["entries"][0]
        audio_url = info.get("url")
        title = info.get("title", "Unknown title")
        return audio_url, title

# --- Play YouTube audio using VLC ---
def play_youtube(mood):
    global player, player_instance, current_title
    query = random.choice(youtube_queries.get(mood, ["relax music"]))
    print(f"[INFO] Searching YouTube for: {query}")

    try:
        audio_url, title = get_youtube_audio_and_title(query)
        if not audio_url:
            print("[ERROR] No audio URL found for query.")
            return

        with player_lock:
            if player_instance is None:
                player_instance = vlc.Instance()

            if player:
                try:
                    player.stop()
                except Exception:
                    pass
                time.sleep(0.15)

            media_player = player_instance.media_player_new()
            media = player_instance.media_new(audio_url)
            media_player.set_media(media)
            media_player.play()

            player = media_player
            current_title = title
            update_gui_safe(current_mood, current_title)

            print(f"[PLAYING] {title}")

    except Exception as e:
        print(f"[ERROR] Could not play from YouTube: {e}")

# --- Mood detection using landmarks (simple heuristic) ---
def detect_mood(landmarks):
    if len(landmarks) < 292:
        return "neutral"

    left = landmarks[61]
    right = landmarks[291]
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]

    mouth_open = abs(top_lip.y - bottom_lip.y) > 0.03
    smile = (right.y < left.y) and mouth_open

    eye = landmarks[33]
    brow = landmarks[70]
    hand_on_head = abs(eye.y - brow.y) < 0.03

    if smile:
        return "happy"
    elif hand_on_head:
        return "sad"
    else:
        return "neutral"

# --- Webcam loop in separate thread ---
def webcam_loop():
    global current_mood
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    while not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                mood = detect_mood(face.landmark)
                if mood != current_mood:
                    current_mood = mood
                    update_gui_safe(current_mood, current_title or "Searching...")
                    threading.Thread(target=play_youtube, args=(mood,), daemon=True).start()
                mp_draw.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_CONTOURS)

        cv2.imshow("Webcam - Press 'q' to quit", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    try:
        root.after(0, root.quit)
    except Exception:
        pass

# --- Thread-safe GUI update ---
def update_gui_safe(mood, title):
    root.after(0, lambda: update_gui(mood, title))

def update_gui(mood, title):
    mood_label.config(text=f"Mood: {mood}")
    song_label.config(text=f"Song: {title}")
    if mood == "happy":
        root.configure(bg="light blue")
    elif mood == "sad":
        root.configure(bg="black")
        mood_label.config(fg="white")
        song_label.config(fg="white")
    else:
        root.configure(bg="white")
        mood_label.config(fg="black")
        song_label.config(fg="black")

# --- Graceful shutdown ---
def on_closing():
    stop_event.set()
    with player_lock:
        if player:
            try:
                player.stop()
            except Exception:
                pass
    time.sleep(0.2)
    try:
        root.destroy()
    except Exception:
        pass

root.protocol("WM_DELETE_WINDOW", on_closing)

t = threading.Thread(target=webcam_loop, daemon=True)
t.start()
root.mainloop()

stop_event.set()
with player_lock:
    if player:
        try:
            player.stop()
        except Exception:
            pass

print("[INFO] Emotion Player exited.")
