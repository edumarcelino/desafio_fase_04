import cv2
import face_recognition
import os
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
import speech_recognition as sr
import math
from transformers import pipeline
import moviepy as mp
import ollama

# Carregar o modelo de sumarização globalmente
summarizer = pipeline("summarization")

# Defina aqui (ou receba como parâmetro) os caminhos:
# Por exemplo:
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO_PATH = os.path.join(SCRIPT_DIR, 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')
OUTPUT_VIDEO_PATH = os.path.join(SCRIPT_DIR, 'output_video_recognize.mp4')
EMOTION_TXT_PATH = os.path.join(SCRIPT_DIR, 'emotions_output.txt')
OUTPUT_AUDIO_PATH = os.path.join(SCRIPT_DIR, 'audio.wav')
TRANSCRIPTION_TXT_PATH = os.path.join(SCRIPT_DIR, 'transcriptions.txt')
TRANSCRIPTION_TIMESTAMP_TXT_PATH = os.path.join(SCRIPT_DIR, 'transcriptions_timestamps.txt')
SUMMARIZE_TXT_PATH = os.path.join(SCRIPT_DIR, 'summarize.txt')
SUMMARIZE_LLM_TXT_PATH = os.path.join(SCRIPT_DIR, 'summarize_llm.txt')
ACTIVITY_TXT_PATH = os.path.join(SCRIPT_DIR, 'activity_log.txt')


def detect_faces(video_path, output_path, emotion_txt_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with open(emotion_txt_path, 'w', encoding='utf-8') as emotion_file:
        emotion_file.write("Frame\tTempo (s)\tEmoção\n")

        for frame_num in tqdm(range(total_frames), desc="Processando vídeo"):
            ret, frame = cap.read()
            if not ret:
                break

            # Análise de emoção com DeepFace (enforce_detection=False evita erro se não achar faces)
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            time_in_sec = frame_num / fps
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Cada "face" do DeepFace pode conter um dicionário com 'dominant_emotion' e 'region'
            for face in result:
                x = face['region']['x']
                y = face['region']['y']
                w = face['region']['w']
                h = face['region']['h']
                dominant_emotion = face['dominant_emotion']

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                emotion_file.write(f"{frame_num}\t{time_in_sec:.2f}\t{dominant_emotion}\n")

            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def extract_audio_from_video(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)


def transcribe_audio(audio_path, text_output_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="en-US")
            print('Transcrição do áudio:', text)
            with open(text_output_path, 'w', encoding='utf-8') as file:
                file.write(text)
        except sr.UnknownValueError:
            print('Google speech recognition could not understand the audio')
        except sr.RequestError as e:
            print(f'Erro na API da Google: {e}')


def transcribe_audio_with_timestamps(audio_path, text_output_path, chunk_duration=10):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        total_duration = source.DURATION
        print(f'Duração total do áudio: {total_duration:.2f} segundos')

        with open(text_output_path, 'w', encoding='utf-8') as file:
            file.write("Início (s)\tFim (s)\tTexto\n")

            for i in range(0, math.ceil(total_duration), chunk_duration):
                audio_chunk = recognizer.record(source, duration=chunk_duration, offset=i)
                try:
                    text = recognizer.recognize_google(audio_chunk, language="en-US")
                    start_time = i
                    end_time = min(i + chunk_duration, total_duration)
                    print(f'Transcrição [{start_time:.2f}s - {end_time:.2f}s]: {text}')
                    file.write(f"{start_time:.2f}\t{end_time:.2f}\t{text}\n")

                except sr.UnknownValueError:
                    print(f'Segmento [{i:.2f}s - {i + chunk_duration:.2f}s]: Não reconhecido')
                except sr.RequestError as e:
                    print(f'Erro na API: {e}')
                    break


def read_txt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo {file_path} não encontrado.")
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def summarize_text(text, max_length=130, min_length=30, do_sample=False, chunk_size=1024):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks:
        effective_max_length = min(len(chunk), max_length)
        summary = summarizer(chunk, max_length=effective_max_length,
                             min_length=min_length, do_sample=do_sample)
        summaries.append(summary[0]['summary_text'])
    return ' '.join(summaries)


def save_summary_to_txt(summary_text, txt_path):
    with open(txt_path, 'w', encoding='utf-8') as file:
        file.write(summary_text)


def send_to_llm(emotions_output, transcriptions_with_timestamps, summarize_path, output_txt_path, movimentos_output):
    with open(emotions_output, 'r', encoding='utf-8') as ef:
        emotions_data = ef.read()
    with open(transcriptions_with_timestamps, 'r', encoding='utf-8') as tf:
        transcriptions_data = tf.read()
    with open(summarize_path, 'r', encoding='utf-8') as sf:
        resumo_data = sf.read()
    with open(movimentos_output, 'r', encoding='utf-8') as mf:
        movimentos_data = mf.read()

    prompt = f"""
    Eu tenho três fontes de dados, que foram extraídas do mesmo vídeo:

    1. **Emoções detectadas no vídeo**:
    {emotions_data}
    
    2. **Transcrições com timestamps**:
    {transcriptions_data}

    3. **Resumo do conteúdo**:
    {resumo_data}

    4. **Movimentos detectados no vídeo**:
    {movimentos_data}

    Com base nessas informações, siga exatamente os três passos a seguir:

    1. **Analise as emoções e os timestamps das transcrições** e faça inferências sobre as ações com base nas emoções.
    
    2. **Analise os movimentos e os timestamps das dos movimentos detectados no video** e faça inferências sobre as ações com base nos movimentos.

    3. **Enriqueça o resumo** utilizando as inferências feitas nas etapas anteriores, integrando as emoções e os movimentos com as transcrições.

    4. **Gere a saída com o resumo enriquecido**, incluindo apenas os seguintes campos:
        - **Título**: Um título conciso e relevante para o conteúdo.
        - **Palavras-chave**: Liste as palavras-chave mais relevantes.
        - **Informações adicionais**: Qualquer dado importante que enriqueça o conteúdo.
        - **Timestamps**: Associe cada parte do resumo com os respectivos timestamps das transcrições.

    A saída deve ser um resumo enriquecido, com base nas fontes de dados fornecidas, sem outros detalhes ou explicações.
    """

    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}],
        )
        enriched_summary = response["message"]["content"]
        with open(output_txt_path, 'w', encoding='utf-8') as output_file:
            output_file.write(enriched_summary)
        print(f"Resultado salvo em {output_txt_path}")
        return enriched_summary
    except Exception as e:
        print(f"Erro ao chamar o Ollama: {e}")
        return None


def run_face_emotion_analysis():
    """
    Função única que chama todos os passos do script,
    na ordem desejada.
    """
    print('1 - INICIANDO A DETECÇÃO DE ROSTOS E EMOÇÕES')
    detect_faces(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, EMOTION_TXT_PATH)

    print('2 - INICIANDO A EXTRAÇÃO DE ÁUDIO DO VÍDEO')
    extract_audio_from_video(INPUT_VIDEO_PATH, OUTPUT_AUDIO_PATH)

    print('3 - TRANSCREVENDO O ÁUDIO PARA TEXTO')
    transcribe_audio(OUTPUT_AUDIO_PATH, TRANSCRIPTION_TXT_PATH)

    print('4 - TRANSCREVENDO O ÁUDIO PARA TEXTO COM OS TIMESTAMPS')
    transcribe_audio_with_timestamps(OUTPUT_AUDIO_PATH, TRANSCRIPTION_TIMESTAMP_TXT_PATH)

    print('5 - CRIAÇÃO DO RESUMO')
    print('5.1 - LEITURA DO TXT GERADO')
    full_text = read_txt(TRANSCRIPTION_TXT_PATH)

    print('5.2 - GERAÇÃO DE RESUMO E SALVA O RESUMO')
    summary = summarize_text(full_text)
    save_summary_to_txt(summary, SUMMARIZE_TXT_PATH)
    print('Resumo salvo em', SUMMARIZE_TXT_PATH)

    print('6 - ENVIANDO PARA A LLM PARA RESPONDER EM LINGUAGEM NATURAL O RESUMO DO VÍDEO')
    send_to_llm(EMOTION_TXT_PATH, TRANSCRIPTION_TIMESTAMP_TXT_PATH, SUMMARIZE_TXT_PATH, SUMMARIZE_LLM_TXT_PATH, ACTIVITY_TXT_PATH)
    
    print('Finalizado o processo de faces, emoções, e LLM.')
