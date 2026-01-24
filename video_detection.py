"""
Object Detection su Video con MobileNetV3 SSD
Processa video frame-by-frame e visualizza bounding boxes
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time
from mobilenet import mobilenetv3_pipeline
from PIL import Image

# Classi COCO (fonte: https://cocodataset.org/)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Colori per le classi (BGR per OpenCV)
np.random.seed(42)
COLORS = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 
          for _ in range(len(COCO_CLASSES))]


def process_video(video_path, output_path=None, conf_threshold=0.5, skip_frames=1):
    """
    Processa video applicando object detection frame-by-frame.
    
    Args:
        video_path (str): Path al video input
        output_path (str, opzionale): Path per salvare video con detection
        conf_threshold (float): Soglia di confidenza per detection
        skip_frames (int): Processa 1 frame ogni N (per velocizzare)
    
    Returns:
        dict: Statistiche processamento (fps, tempo, num_frames, ecc.)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")
    
    # Apri video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire video: {video_path}")
    
    # Info video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*70}")
    print(f"VIDEO INFO")
    print(f"{'='*70}")
    print(f"  File: {video_path.name}")
    print(f"  Risoluzione: {width}x{height}")
    print(f"  FPS originale: {fps}")
    print(f"  Frame totali: {total_frames}")
    print(f"  Durata: {total_frames/fps:.2f}s")
    print(f"  Skip frames: 1 ogni {skip_frames}")
    print(f"{'='*70}\n")
    
    # Setup output video se richiesto
    writer = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps//skip_frames, (width, height))
    
    # Rileva device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Statistiche
    frame_count = 0
    processed_count = 0
    total_detections = 0
    processing_times = []
    
    # Progress bar
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            pbar.update(1)
            
            # Skip frames per velocizzare
            if frame_count % skip_frames != 0:
                if writer:
                    writer.write(frame)
                continue
            
            processed_count += 1
            
            # Converti BGR (OpenCV) -> RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Object detection
            start_time = time.time()
            results = mobilenetv3_pipeline(pil_image, device=device, conf_threshold=conf_threshold)
            detection_time = (time.time() - start_time) * 1000  # ms
            processing_times.append(detection_time)
            
            # Disegna bounding boxes
            num_detections = results['num_detections']
            total_detections += num_detections
            
            for i in range(num_detections):
                # Coordinate box [x1, y1, x2, y2]
                x1, y1, x2, y2 = results['boxes'][i].astype(int)
                
                # Clip coordinate ai bordi dell'immagine
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # Label e score
                label_idx = int(results['labels'][i])
                class_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f"class_{label_idx}"
                score = results['scores'][i]
                
                # Colore per questa classe
                color = COLORS[label_idx % len(COLORS)]
                
                # Disegna rettangolo
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Aggiungi label con background
                label_text = f"{class_name}: {score:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Box per il testo
                cv2.rectangle(frame, 
                            (x1, y1 - text_height - baseline - 5), 
                            (x1 + text_width, y1), 
                            color, -1)
                
                # Testo
                cv2.putText(frame, label_text, 
                          (x1, y1 - baseline - 2),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Info frame (angolo in alto a sinistra)
            info_text = f"Frame: {frame_count}/{total_frames} | Objects: {num_detections} | {detection_time:.1f}ms"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Salva frame
            if writer:
                writer.write(frame)
        
    finally:
        pbar.close()
        cap.release()
        if writer:
            writer.release()
    
    # Calcola statistiche
    avg_detection_time = np.mean(processing_times) if processing_times else 0
    avg_fps = 1000 / avg_detection_time if avg_detection_time > 0 else 0
    
    stats = {
        'total_frames': frame_count,
        'processed_frames': processed_count,
        'total_detections': total_detections,
        'avg_detections_per_frame': total_detections / processed_count if processed_count > 0 else 0,
        'avg_detection_time_ms': avg_detection_time,
        'avg_fps': avg_fps,
        'video_fps': fps,
        'realtime_capable': avg_fps >= fps,
    }
    
    # Stampa report
    print(f"\n{'='*70}")
    print(f"PROCESSING REPORT")
    print(f"{'='*70}")
    print(f"  Frame processati: {processed_count}/{frame_count}")
    print(f"  Oggetti rilevati: {total_detections}")
    print(f"  Media oggetti/frame: {stats['avg_detections_per_frame']:.1f}")
    print(f"  Tempo detection medio: {avg_detection_time:.1f} ms")
    print(f"  FPS medio processing: {avg_fps:.1f}")
    print(f"  FPS video originale: {fps}")
    
    if stats['realtime_capable']:
        print(f"  Realtime capable: {avg_fps:.1f} >= {fps} fps")
    else:
        print(f"  Non realtime: {avg_fps:.1f} < {fps} fps")
        print(f"  Suggerimento: usa skip_frames={int(fps/avg_fps) + 1} per realtime")
    
    if output_path:
        print(f"\n  Video salvato: {output_path}")
        print(f"     Dimensione: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    print(f"{'='*70}\n")
    
    return stats


def play_video_with_detection(video_path, conf_threshold=0.5, skip_frames=1):
    """
    Riproduce video con object detection in tempo reale.
    Premi 'q' per uscire, 'p' per pausa.
    
    Args:
        video_path (str): Path al video
        conf_threshold (float): Soglia confidenza
        skip_frames (int): Processa 1 frame ogni N
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nRiproduzione video con detection")
    print(f"Device: {device.upper()}")
    print(f"Comandi: 'q' = quit, 'p' = pause, SPACE = pause\n")
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nVideo terminato")
                break
            
            frame_count += 1
            
            # Processa solo alcuni frame
            if frame_count % skip_frames == 0:
                # Detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                start = time.time()
                results = mobilenetv3_pipeline(pil_image, device=device, conf_threshold=conf_threshold)
                detection_time = (time.time() - start) * 1000
                
                # Disegna boxes
                for i in range(results['num_detections']):
                    x1, y1, x2, y2 = results['boxes'][i].astype(int)
                    label_idx = int(results['labels'][i])
                    class_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f"class_{label_idx}"
                    score = results['scores'][i]
                    color = COLORS[label_idx % len(COLORS)]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {score:.2f}"
                    cv2.putText(frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Info
                info = f"Frame: {frame_count} | Objects: {results['num_detections']} | {detection_time:.0f}ms"
                cv2.putText(frame, info, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostra frame
        cv2.imshow('Object Detection - MobileNetV3 SSD', frame)
        
        # Gestisci input
        key = cv2.waitKey(1000//fps) & 0xFF
        if key == ord('q'):
            print("\nInterrotto dall'utente")
            break
        elif key == ord('p') or key == ord(' '):
            paused = not paused
            status = "PAUSA" if paused else "PLAY"
            print(f"\r{status}", end='', flush=True)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    # Path al video
    video_file = "data/mobilenet/videos/sample.mp4"
    
    if not Path(video_file).exists():
        print(f"Video non trovato: {video_file}")
        print("Scarica un video con: python download-video-sample.py")
        sys.exit(1)
    
    # Processa e salva video con detection
    output_file = "data/mobilenet/videos/sample_detected.mp4"
    
    print("\nOPZIONI:")
    print("1. Processa e salva video")
    print("2. Riproduzione live con detection")
    print("3. Entrambi\n")
    
    choice = input("Scelta (1/2/3): ").strip() or "3"
    
    if choice in ["1", "3"]:
        print("\nProcessamento video...")
        stats = process_video(
            video_file, 
            output_path=output_file,
            conf_threshold=0.5,
            skip_frames=1  # Processa tutti i frame
        )
    
    if choice in ["2", "3"]:
        print("\nAvvio riproduzione...")
        # Usa skip_frames se non è realtime capable
        skip = 2 if stats.get('avg_fps', 30) < 15 else 1
        play_video_with_detection(video_file, conf_threshold=0.5, skip_frames=skip)
