from flask import Flask, render_template, Response, request, redirect
import cv2, os, numpy as np
from PIL import Image
import time

app = Flask(__name__)

# Inisialisasi detector & recognizer
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# State untuk perekaman terkontrol
capture_requested = False

# ====== HALAMAN UTAMA ======
@app.route('/')
def index():
    return render_template('index.html')


# ====== REGISTER (DAFTAR WAJAH) ======
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        global current_name, current_id
        current_name = request.form['name']
        current_id = request.form['id']
        return redirect('/register_camera')
    return render_template('daftar.html')


@app.route('/register_camera')
def register_camera():
    return render_template('register_camera.html')

# Trigger dari tombol untuk mulai ambil gambar sesuai instruksi
@app.route('/capture', methods=['POST'])
def trigger_capture():
    global capture_requested
    capture_requested = True
    return "OK"


# ====== STREAM KAMERA UNTUK PENDAFTARAN ======
def gen_register():
    global capture_requested
    cam = cv2.VideoCapture(0)
    count = 0

    # Target per instruksi dan kontrol
    per_instruction_target = 3
    instruction_index = 0
    captured_for_instruction = 0
    stable_frames = 0
    last_save_time = 0
    min_interval = 0.5  # detik

    # Instruksi untuk variasi wajah
    instructions = [
        "Wajah normal ke depan",
        "Senyum lebar",
        "Wajah datar/serius",
        "Sedikit cemberut",
        "Putar kepala ke kiri",
        "Putar kepala ke kanan",
        "Angkat dagu sedikit",
        "Turunkan dagu sedikit"
    ]

    while True:
        success, frame = cam.read()
        if not success:
            break

        # Mirror (flip horizontal)
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # Stabilitas deteksi wajah
        if len(faces) > 0:
            stable_frames += 1
        else:
            stable_frames = 0

        # Instruksi saat ini
        current_instruction = instructions[instruction_index] if instruction_index < len(instructions) else "Selesai!"

        # Hanya simpan jika user menekan tombol dan wajah stabil
        should_save = capture_requested and stable_frames >= 5 and instruction_index < len(instructions)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            if should_save and (time.time() - last_save_time) > min_interval and captured_for_instruction < per_instruction_target:
                count += 1
                cv2.putText(frame, f"Rekam: {count}/{per_instruction_target*len(instructions)}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imwrite(f"static/wajah/{current_id}_{current_name}_{count}.jpg", gray[y:y+h, x:x+w])
                captured_for_instruction += 1
                last_save_time = time.time()

                # Jika sudah cukup untuk instruksi ini, lanjut ke instruksi berikutnya
                if captured_for_instruction >= per_instruction_target:
                    capture_requested = False
                    instruction_index += 1
                    captured_for_instruction = 0
                    stable_frames = 0

        # Overlay instruksi dan progres
        total_target = per_instruction_target * len(instructions)
        cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Instruksi: {current_instruction}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status_text = "Klik tombol 'Ambil' saat sudah sesuai instruksi" if not capture_requested else "Mengambil... tahan pose"
        cv2.putText(frame, status_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2)

        # Tampilkan frame di browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Berhenti otomatis setelah target terpenuhi
        if count >= total_target or instruction_index >= len(instructions):
            break

    cam.release()
    cv2.destroyAllWindows()


@app.route('/video_register')
def video_register():
    return Response(gen_register(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ====== TRAINING DATA ======
@app.route('/train')
def train():
    path = 'static/wajah'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples, ids = [], []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split('_')[0])
        faces = face_detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    face_recognizer.train(face_samples, np.array(ids))
    os.makedirs('model', exist_ok=True)
    face_recognizer.write('model/training.xml')
    return "✅ Training selesai dan disimpan di model/training.xml"


# ====== DETEKSI (UJIAN) ======
@app.route('/exam')
def exam():
    return render_template('exam.html')


def gen_exam():
    face_recognizer.read('model/training.xml')
    camera = cv2.VideoCapture(0)

    total_frames = 0
    detected_frames = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Mirror (flip horizontal)
        frame = cv2.flip(frame, 1)

        total_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5)

        if len(faces) > 0:
            detected_frames += 1

        for (x, y, w, h) in faces:
            id, conf = face_recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 60:
                label = f"ID: {id}"
                warna = (0, 255, 0)
            else:
                label = "Tidak Dikenal"
                warna = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), warna, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, warna, 2)

        # Overlay persentase deteksi di pojok kiri atas
        percent = int((detected_frames / total_frames) * 100) if total_frames > 0 else 0
        cv2.rectangle(frame, (10, 10), (180, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"{percent}% terdeteksi", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()
    cv2.destroyAllWindows()


@app.route('/video_exam')
def video_exam():
    return Response(gen_exam(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ====== MAIN PROGRAM ======
if __name__ == '__main__':
    os.makedirs("static/wajah", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    app.run(debug=True)
