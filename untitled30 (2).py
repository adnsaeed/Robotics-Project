import cv2
import numpy as np
import os
import io
import PIL
from base64 import b64decode, b64encode
from google.colab import drive
from google.colab.output import eval_js
from IPython.display import display, Javascript, Image
!pip install face_recognition
import face_recognition

drive.mount('/content/drive')

def encode_faces(directory):
    encodings = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        image = face_recognition.load_image_file(filepath)
        face_enc = face_recognition.face_encodings(image)
        if face_enc:
            encodings.append(face_enc[0])
    return encodings

adnan_dir = '/content/drive/MyDrive/Dataset/Adnan'
adnan_encodings = encode_faces(adnan_dir)
print(f"Adnan encodings loaded: {len(adnan_encodings)}")
known_face_encodings = adnan_encodings
known_face_names = ["Adnan"] * len(adnan_encodings)

def decode_js_image(js_reply):
    image_data = b64decode(js_reply.split(',')[1])
    np_img = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(np_img, flags=1)
    return img

def convert_bbox_to_bytes(bbox_array):
    bbox_image = PIL.Image.fromarray(bbox_array, 'RGBA')
    buffer = io.BytesIO()
    bbox_image.save(buffer, format='png')
    bbox_data = b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{bbox_data}'

def stream_video():
    js_code = '''
        var video;
        var div;
        var stream;
        var captureCanvas;
        var pendingResolve;
        var shutdown = false;

        function removeElements() {
            stream.getTracks().forEach(track => track.stop());
            video.remove();
            div.remove();
        }

        function onFrame() {
            if (!shutdown) {
                window.requestAnimationFrame(onFrame);
            }
            if (pendingResolve) {
                captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
                var result = captureCanvas.toDataURL('image/jpeg', 0.8);
                pendingResolve(result);
                pendingResolve = null;
            }
        }

        async function createDom() {
            if (div) return stream;
            div = document.createElement('div');
            document.body.appendChild(div);

            video = document.createElement('video');
            video.width = 640;
            video.onclick = () => { shutdown = true; };
            stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            await video.play();
            div.appendChild(video);

            captureCanvas = document.createElement('canvas');
            captureCanvas.width = 640;
            captureCanvas.height = 480;
            window.requestAnimationFrame(onFrame);

            return stream;
        }

        async function streamFrame() {
            if (shutdown) {
                removeElements();
                return '';
            }
            stream = await createDom();
            return await new Promise(resolve => { pendingResolve = resolve; });
        }
    '''
    display(Javascript(js_code))

def get_video_frame():
    data = eval_js('streamFrame()')
    return data

smtp_server = 'smtp.office365.com'
smtp_port = 587
email_user = 'ProjectRobotics@hotmail.com'
email_password = 'q1w2e3r4t5'
email_to = 'adn20200976@std.psut.edu.jo'

def send_alert_email(image):
    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_to
    msg['Subject'] = 'Threat! unknown person is here!'

    msg.attach(MIMEText('Threat! unknown person is here!', 'plain'))

    img_data = cv2.imencode('.jpg', image)[1].tobytes()
    image = MIMEImage(img_data, name='unknown_face.jpg')
    msg.attach(image)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_user, email_password)
            server.sendmail(email_user, email_to, msg.as_string())
            print('Email sent successfully')
    except Exception as e:
        print(f"Failed to send email: {e}")

face_distance_threshold = 0.65

stream_video()
bbox = ''

while True:
    try:
        js_reply = get_video_frame()
        if not js_reply:
            break

        img = decode_js_image(js_reply)
        bbox_array = np.zeros((480, 640, 4), dtype=np.uint8)

        face_locs = face_recognition.face_locations(img)
        face_encs = face_recognition.face_encodings(img, face_locs)

        for (top, right, bottom, left), face_enc in zip(face_locs, face_encs):
            name = "Unknown"
            face_dists = face_recognition.face_distance(known_face_encodings, face_enc)
            best_match_idx = np.argmin(face_dists)

            if face_dists[best_match_idx] < face_distance_threshold:
                name = known_face_names[best_match_idx]

            cv2.rectangle(bbox_array, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(bbox_array, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if name == "Unknown":
                send_alert_email(img)

        bbox_array[:, :, 3] = (bbox_array.max(axis=2) > 0).astype(int) * 255
        bbox = convert_bbox_to_bytes(bbox_array)
    except Exception as e:
        print(f"An error occurred: {e}")
        break
