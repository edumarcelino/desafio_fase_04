import math
import mediapipe as mp
from movimentos_anomalos import log_movimento_anomalo

def detect_hand_touching_face(results, threshold=0.03):
    """
    Retorna True se a mão (esquerda ou direita) está tocando o rosto (distância 2D < threshold).
    """
    face_landmarks = results.face_landmarks
    if not face_landmarks:
        return False

    # Converte rosto para lista (x, y)
    face_points = [(lm.x, lm.y) for lm in face_landmarks.landmark]

    left_hand_landmarks = results.left_hand_landmarks
    right_hand_landmarks = results.right_hand_landmarks

    def dist_2d(x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # IDs das pontas dos dedos (MediaPipe Hands)
    finger_tips = [4, 8, 12, 16, 20]

    # Verifica mão esquerda
    if left_hand_landmarks:
        for tip_id in finger_tips:
            tip = left_hand_landmarks.landmark[tip_id]
            for (fx, fy) in face_points:
                if dist_2d(tip.x, tip.y, fx, fy) < threshold:
                    return True

    # Verifica mão direita
    if right_hand_landmarks:
        for tip_id in finger_tips:
            tip = right_hand_landmarks.landmark[tip_id]
            for (fx, fy) in face_points:
                if dist_2d(tip.x, tip.y, fx, fy) < threshold:
                    return True

    return False


def handle_face_touch_event(
    frame_idx, fps,
    touching_face,
    face_touch_state,
    csv_writer,
    face_touch_count_dict,
    anomaly_count_dict
):
    """
    Controla a lógica de "mão tocando o rosto" por >= 0.8s.
    Se o evento terminar e for >= 0.8s, registra "Mãos tocando o rosto".
    Caso contrário, registra "Movimento Anômalo".
    
    face_touch_state: {
      'currently_in_event': False,
      'touch_frames_count': 0,
      'touch_min_frames': int(0.8 * fps)
    }
    face_touch_count_dict: {'count': 0}
    anomaly_count_dict: {'count': 0}
    """
    if touching_face:
        if not face_touch_state['currently_in_event']:
            # Começa novo evento de toque
            face_touch_state['currently_in_event'] = True
            face_touch_state['touch_frames_count'] = 1
        else:
            face_touch_state['touch_frames_count'] += 1
    else:
        # Se não está tocando agora, mas estava em evento
        if face_touch_state['currently_in_event']:
            # Evento terminou, checa duração
            if face_touch_state['touch_frames_count'] >= face_touch_state['touch_min_frames']:
                # Toque longo (>= 0.4s)
                time_in_sec = frame_idx / fps
                csv_writer.writerow([f"{time_in_sec:.2f}", "Mãos tocando o rosto"])
                face_touch_count_dict['count'] += 1
            else:
                # Toque rápido (< 0.4s) => movimento anômalo
                log_movimento_anomalo(frame_idx, fps, csv_writer, anomaly_count_dict)

            # Reseta estado
            face_touch_state['currently_in_event'] = False
            face_touch_state['touch_frames_count'] = 0
