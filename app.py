from flask import Flask, render_template, Response, request, redirect
import cv2, os, numpy as np
from PIL import Image

app = Flask(__name__)

# Inisialisasi detector & recognizer
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

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


# ====== STREAM KAMERA UNTUK PENDAFTARAN ======
def gen_register():
    cam = cv2.VideoCapture(0)
    count = 0
    while True:
        success, frame = cam.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, f"Rekam: {count}/30", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Simpan hasil wajah ke folder
            cv2.imwrite(f"static/wajah/{current_id}_{current_name}_{count}.jpg", gray[y:y+h, x:x+w])

        # Tampilkan frame di browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Berhenti otomatis setelah 30 wajah
        if count >= 30:
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

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5)

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
