import os
from detect_pose import detect_pose_with_holistic

# Importamos a função do arquivo detect_faces_emotions.py
from desafio_fase_4 import run_face_emotion_analysis

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Caminhos do seu outro script
    input_video_path = os.path.join(script_dir, 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')
    output_video_path = os.path.join(script_dir, 'output_video_pose_detect_holistic.mp4')

    print("=== Iniciando detecção de movimentos e poses (braços, rosto, etc.) ===")
    detect_pose_with_holistic(input_video_path, output_video_path)

    print("\n=== Iniciando detecção de faces, emoções e processamentos adicionais ===")
    run_face_emotion_analysis()

    print("=== Tudo concluído! ===")
