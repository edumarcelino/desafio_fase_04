import cv2
import mediapipe as mp
import os
from tqdm import tqdm

from is_arm_up import arms_up_in_frame, handle_arm_up_event
from is_hand_touching_face import detect_hand_touching_face, handle_face_touch_event

def detect_pose_with_holistic(video_path, output_path):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo:", video_path)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Em vez de CSV, vamos gravar em um arquivo TXT
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(script_dir, 'activity_log.txt')
    
    # Abre o arquivo TXT para escrita
    txt_file = open(txt_path, 'w', encoding='utf-8')
    # Escreve o cabeçalho com 3 colunas
    txt_file.write("Frame\tTempo (s)\tAtividade\n")

    # Estados para braços (>= 0.3s)
    arm_up_state = {
        'currently_in_event': False,
        'frames_count': 0,
        'min_frames': int(0.3 * fps)  # 0.3 segundos
    }
    arm_count_dict = {'count': 0}

    # Estados para toque no rosto (>= 0.4s)
    face_touch_state = {
        'currently_in_event': False,
        'touch_frames_count': 0,
        'touch_min_frames': int(0.4 * fps)
    }
    face_touch_count_dict = {'count': 0}
    anomaly_count_dict = {'count': 0}

    for frame_idx in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # 1) Desenhar landmarks
        # Pose
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            )
        # Face
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1),
            )
        # Mão esquerda
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
            )
        # Mão direita
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
            )

        # 2) Braços levantados (>= 0.3s)
        if results.pose_landmarks:
            arms_up_now = arms_up_in_frame(results.pose_landmarks.landmark)
            handle_arm_up_event_custom(
                frame_idx=frame_idx,
                fps=fps,
                arms_up_this_frame=arms_up_now,
                arm_up_state=arm_up_state,
                txt_file=txt_file,             # passamos o arquivo txt
                arm_count_dict=arm_count_dict
            )

        # 3) Mãos tocando o rosto / movimento anômalo
        touching_face = detect_hand_touching_face(results, threshold=0.03)
        handle_face_touch_event_custom(
            frame_idx=frame_idx,
            fps=fps,
            touching_face=touching_face,
            face_touch_state=face_touch_state,
            txt_file=txt_file,                # passamos o arquivo txt
            face_touch_count_dict=face_touch_count_dict,
            anomaly_count_dict=anomaly_count_dict
        )

        # 4) Exibir contadores na tela
        cv2.putText(
            frame,
            f'Movimento dos bracos: {arm_count_dict["count"]}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )

        cv2.putText(
            frame,
            f'Mao no rosto (>=0.4s): {face_touch_count_dict["count"]}',
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )

        cv2.putText(
            frame,
            f'Mov. anomalo (<0.4s): {anomaly_count_dict["count"]}',
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )

        # 5) Salva o frame no vídeo de saída
        out.write(frame)

        # 6) Opcional: exibe na tela
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Se terminar o vídeo e ainda estivermos no meio de um "evento" de braços
    if arm_up_state['currently_in_event']:
        if arm_up_state['frames_count'] >= arm_up_state['min_frames']:
            time_in_sec = total_frames / fps
            # Registramos no TXT
            txt_file.write(f"{total_frames}\t{time_in_sec:.2f}\tMovimento dos Braços\n")
            arm_count_dict['count'] += 1

    # ... e evento de "mãos tocando o rosto"
    if face_touch_state['currently_in_event']:
        if face_touch_state['touch_frames_count'] >= face_touch_state['touch_min_frames']:
            time_in_sec = total_frames / fps
            txt_file.write(f"{total_frames}\t{time_in_sec:.2f}\tMãos tocando o rosto\n")
            face_touch_count_dict['count'] += 1
        else:
            time_in_sec = total_frames / fps
            txt_file.write(f"{total_frames}\t{time_in_sec:.2f}\tMovimento Anomalo\n")
            anomaly_count_dict['count'] += 1

    cap.release()
    out.release()
    txt_file.close()
    cv2.destroyAllWindows()


def handle_arm_up_event_custom(frame_idx, fps, arms_up_this_frame, arm_up_state, txt_file, arm_count_dict):
    """
    Versão customizada que escreve no arquivo TXT em vez de CSV.
    """
    if arms_up_this_frame:
        if not arm_up_state['currently_in_event']:
            arm_up_state['currently_in_event'] = True
            arm_up_state['frames_count'] = 1
        else:
            arm_up_state['frames_count'] += 1
    else:
        # Se os braços abaixaram agora e estávamos em evento
        if arm_up_state['currently_in_event']:
            if arm_up_state['frames_count'] >= arm_up_state['min_frames']:
                time_in_sec = frame_idx / fps
                txt_file.write(f"{frame_idx}\t{time_in_sec:.2f}\tMovimento dos Braços\n")
                arm_count_dict['count'] += 1

            arm_up_state['currently_in_event'] = False
            arm_up_state['frames_count'] = 0


def handle_face_touch_event_custom(frame_idx, fps, touching_face, face_touch_state, txt_file, face_touch_count_dict, anomaly_count_dict):
    """
    Versão customizada que escreve no arquivo TXT em vez de CSV.
    """
    if touching_face:
        if not face_touch_state['currently_in_event']:
            face_touch_state['currently_in_event'] = True
            face_touch_state['touch_frames_count'] = 1
        else:
            face_touch_state['touch_frames_count'] += 1
    else:
        if face_touch_state['currently_in_event']:
            # Verifica duração
            if face_touch_state['touch_frames_count'] >= face_touch_state['touch_min_frames']:
                time_in_sec = frame_idx / fps
                txt_file.write(f"{frame_idx}\t{time_in_sec:.2f}\tMãos tocando o rosto\n")
                face_touch_count_dict['count'] += 1
            else:
                time_in_sec = frame_idx / fps
                txt_file.write(f"{frame_idx}\t{time_in_sec:.2f}\tMovimento Anomalo\n")
                anomaly_count_dict['count'] += 1

            face_touch_state['currently_in_event'] = False
            face_touch_state['touch_frames_count'] = 0
