def log_movimento_anomalo(frame_idx, fps, csv_writer, anomaly_count_dict):
    """
    Escreve no CSV (tempo, 'Movimento Anomalo')
    e incrementa o contador de anomalias.
    """
    time_in_sec = frame_idx / fps
    csv_writer.writerow([f"{time_in_sec:.2f}", "Movimento Anomalo"])
    anomaly_count_dict['count'] += 1
