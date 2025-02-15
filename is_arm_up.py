import mediapipe as mp

def arms_up_in_frame(pose_landmarks):
    """
    Retorna True se AMBOS os braços estiverem levantados neste frame.
    Caso contrário, retorna False.
    """
    mp_holistic = mp.solutions.holistic

    if not pose_landmarks:
        return False

    # IDs relevantes
    left_eye_id = mp_holistic.PoseLandmark.LEFT_EYE.value
    right_eye_id = mp_holistic.PoseLandmark.RIGHT_EYE.value
    left_elbow_id = mp_holistic.PoseLandmark.LEFT_ELBOW.value
    right_elbow_id = mp_holistic.PoseLandmark.RIGHT_ELBOW.value

    left_eye = pose_landmarks[left_eye_id]
    right_eye = pose_landmarks[right_eye_id]
    left_elbow = pose_landmarks[left_elbow_id]
    right_elbow = pose_landmarks[right_elbow_id]

    # Considera "braço para cima" se o cotovelo estiver
    # em Y menor que o olho (lembre-se: y=0 é topo da imagem).
    left_arm_up = (left_elbow.y < left_eye.y)
    right_arm_up = (right_elbow.y < right_eye.y)

    return (left_arm_up and right_arm_up)


def handle_arm_up_event(frame_idx, fps, arms_up_this_frame, arm_up_state, csv_writer, arm_count_dict):
    """
    Controla a lógica de "braços levantados" por >= 0.3s.
    Se o evento terminar e a duração for >= 0.3s, grava no CSV.
    
    arm_up_state: {
        'currently_in_event': False,
        'frames_count': 0,
        'min_frames': int(0.3 * fps)
    }
    arm_count_dict: {'count': 0}  # contador de movimentos de braço
    """
    if arms_up_this_frame:
        # Braços levantados neste frame
        if not arm_up_state['currently_in_event']:
            # Inicia um novo "evento" de braços levantados
            arm_up_state['currently_in_event'] = True
            arm_up_state['frames_count'] = 1
        else:
            # Já estava em evento, incrementa
            arm_up_state['frames_count'] += 1
    else:
        # Braços não estão levantados neste frame
        if arm_up_state['currently_in_event']:
            # Evento acabou de terminar
            if arm_up_state['frames_count'] >= arm_up_state['min_frames']:
                # Foi um movimento de braços que durou >= 0.3s
                time_in_sec = frame_idx / fps
                csv_writer.writerow([f"{time_in_sec:.2f}", "Movimento dos Braços"])
                arm_count_dict['count'] += 1

            # Reseta estado
            arm_up_state['currently_in_event'] = False
            arm_up_state['frames_count'] = 0
