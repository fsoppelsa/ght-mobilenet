"""
mobilenet_failure.py - Funzioni per l'analisi dei failure cases di MobileNetV3.

Modulo di supporto per il notebook di analisi dei failure cases.
Contiene funzioni per inference, valutazione e visualizzazione.

UniPA - Visione Artificiale A.A. 2025/2026
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from tqdm import tqdm
from collections import defaultdict
import warnings

# Classi COCO (91 ID ma 80 classi effettive per detection)
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
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Coppie di classi comunemente confuse
CONFUSED_PAIRS = [
    ('cat', 'dog'), ('chair', 'couch'), ('car', 'truck'), ('cup', 'bowl'),
    ('knife', 'fork'), ('laptop', 'tv'), ('bed', 'couch'), ('bench', 'chair')
]


def load_detection_model(model_name: str = 'mobilenet_v3_large', 
                         device: str = None) -> Tuple[torch.nn.Module, str]:
    """
    Carica un modello di object detection pre-trained.
    
    Args:
        model_name: 'mobilenet_v3_large', 'mobilenet_v3_large_320', 'resnet50'
        device: 'cuda', 'cpu' o None per auto-detect
    
    Returns:
        Tuple (modello, device_usato)
    """
    # Auto-detect device con fallback sicuro
    if device is None:
        if torch.cuda.is_available():
            try:
                # Test allocazione GPU
                torch.cuda.empty_cache()
                _ = torch.zeros(1).cuda()
                device = 'cuda'
            except RuntimeError:
                print("CUDA disponibile ma non funzionante, uso CPU")
                device = 'cpu'
        else:
            device = 'cpu'
    
    print(f"Caricamento {model_name} su {device}...")
    
    if model_name == 'ssdlite':
        # SSDLite320 con MobileNetV3-Large backbone (~3.4M params)
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = ssdlite320_mobilenet_v3_large(weights=weights)
    elif model_name == 'mobilenet_v3_large':
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    elif model_name == 'mobilenet_v3_large_320':
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    elif model_name == 'resnet50':
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
    else:
        raise ValueError(f"Modello non supportato: {model_name}")
    
    # Caricamento con fallback a CPU
    try:
        model = model.to(device)
    except RuntimeError as e:
        print(f"Errore GPU: {e}")
        print("Fallback a CPU...")
        device = 'cpu'
        model = model.to(device)
    
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Modello caricato: {n_params:.1f}M parametri su {device}")
    
    return model, device


def preprocess_image(image_path: str, device: str) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocessa un'immagine per l'inference.
    
    Returns:
        Tuple (tensore, immagine_rgb)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img_rgb).to(device)
    
    return img_tensor, img_rgb


def run_inference(model: torch.nn.Module, image_tensor: torch.Tensor, 
                  threshold: float = 0.5) -> Dict[str, Any]:
    """
    Esegue inference su un'immagine.
    
    Args:
        model: Modello detection
        image_tensor: Tensore immagine
        threshold: Soglia confidence minima
    
    Returns:
        Dict con boxes, labels, scores filtrati
    """
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    
    mask = predictions['scores'] >= threshold
    
    return {
        'boxes': predictions['boxes'][mask].cpu().numpy(),
        'labels': predictions['labels'][mask].cpu().numpy(),
        'scores': predictions['scores'][mask].cpu().numpy(),
        'all_boxes': predictions['boxes'].cpu().numpy(),
        'all_labels': predictions['labels'].cpu().numpy(),
        'all_scores': predictions['scores'].cpu().numpy()
    }


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calcola IoU tra due bounding box [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def get_box_dimensions(box: np.ndarray) -> Tuple[float, float, float]:
    """Restituisce (width, height, area) di un box."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    return w, h, w * h


def classify_size(area: float) -> str:
    """Classifica la dimensione di un oggetto."""
    if area < 32 * 32:
        return 'small'
    elif area < 96 * 96:
        return 'medium'
    return 'large'


def classify_failures(prediction: Dict, ground_truth: Dict, 
                      iou_threshold: float = 0.5) -> List[Dict]:
    """
    Classifica i tipi di failure per una singola immagine.
    
    Tipi di failure:
    - false_negative: GT presente ma non rilevato
    - false_positive: Detection senza GT corrispondente
    - wrong_class: Classe predetta diversa da GT
    - poor_localization: IoU < threshold ma oggetto rilevato
    
    Returns:
        Lista di failure con tipo e dettagli
    """
    failures = []
    pred_boxes = prediction.get('boxes', np.array([]))
    pred_labels = prediction.get('labels', np.array([]))
    pred_scores = prediction.get('scores', np.array([]))
    
    gt_boxes = ground_truth.get('boxes', np.array([]))
    gt_labels = ground_truth.get('labels', np.array([]))
    
    if len(gt_boxes) == 0:
        gt_boxes = np.array([]).reshape(0, 4)
    if len(gt_labels) == 0:
        gt_labels = np.array([])
    
    matched_gt = set()
    matched_pred = set()
    
    # Match predizioni con GT
    for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_label = gt_labels[best_gt_idx]
            matched_gt.add(best_gt_idx)
            matched_pred.add(i)
            
            if pred_label != gt_label:
                failures.append({
                    'type': 'wrong_class',
                    'pred_label': int(pred_label),
                    'gt_label': int(gt_label),
                    'pred_class': COCO_CLASSES[pred_label] if pred_label < len(COCO_CLASSES) else 'unknown',
                    'gt_class': COCO_CLASSES[gt_label] if gt_label < len(COCO_CLASSES) else 'unknown',
                    'confidence': float(pred_score),
                    'iou': float(best_iou),
                    'box': pred_box.tolist()
                })
        elif best_iou > 0.1 and best_iou < iou_threshold:
            failures.append({
                'type': 'poor_localization',
                'pred_label': int(pred_label),
                'pred_class': COCO_CLASSES[pred_label] if pred_label < len(COCO_CLASSES) else 'unknown',
                'confidence': float(pred_score),
                'iou': float(best_iou),
                'box': pred_box.tolist()
            })
            matched_pred.add(i)
        else:
            failures.append({
                'type': 'false_positive',
                'pred_label': int(pred_label),
                'pred_class': COCO_CLASSES[pred_label] if pred_label < len(COCO_CLASSES) else 'unknown',
                'confidence': float(pred_score),
                'box': pred_box.tolist()
            })
    
    # False negatives
    for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        if j not in matched_gt:
            w, h, area = get_box_dimensions(gt_box)
            failures.append({
                'type': 'false_negative',
                'gt_label': int(gt_label),
                'gt_class': COCO_CLASSES[gt_label] if gt_label < len(COCO_CLASSES) else 'unknown',
                'box': gt_box.tolist(),
                'size_category': classify_size(area),
                'area': float(area)
            })
    
    return failures


def analyze_image_conditions(image: np.ndarray) -> Dict[str, Any]:
    """
    Analizza le condizioni dell'immagine che potrebbero causare failure.
    
    Returns:
        Dict con indicatori di condizioni problematiche
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Illuminazione
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Blur detection (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Contrasto
    contrast = gray.max() - gray.min()
    
    return {
        'brightness': float(mean_brightness),
        'brightness_std': float(std_brightness),
        'is_dark': mean_brightness < 50,
        'is_bright': mean_brightness > 200,
        'laplacian_var': float(laplacian_var),
        'is_blurry': laplacian_var < 100,
        'contrast': float(contrast),
        'is_low_contrast': contrast < 100
    }


def categorize_failure_reason(failure: Dict, image: np.ndarray, 
                              all_gt_boxes: np.ndarray) -> str:
    """
    Determina la categoria di failure basandosi su caratteristiche.
    
    Categorie:
    - small_object: Oggetto troppo piccolo
    - occlusion: Possibile occlusione
    - lighting: Illuminazione estrema
    - overlap: Oggetti sovrapposti
    - confused_class: Classi simili confuse
    - blur: Immagine sfocata
    - edge_object: Oggetto ai bordi
    """
    conditions = analyze_image_conditions(image)
    
    # Verifica condizioni immagine
    if conditions['is_blurry']:
        return 'blur'
    if conditions['is_dark'] or conditions['is_bright']:
        return 'lighting'
    
    # Verifica tipo di failure
    if failure['type'] == 'false_negative':
        if failure.get('size_category') == 'small':
            return 'small_object'
        
        # Verifica se ai bordi
        box = failure.get('box', [0, 0, 0, 0])
        h, w = image.shape[:2]
        if box[0] < 10 or box[1] < 10 or box[2] > w - 10 or box[3] > h - 10:
            return 'edge_object'
        
        # Verifica overlap con altri GT
        for gt_box in all_gt_boxes:
            if not np.array_equal(gt_box, box):
                iou = calculate_iou(np.array(box), gt_box)
                if iou > 0.3:
                    return 'overlap'
    
    if failure['type'] == 'wrong_class':
        pred_class = failure.get('pred_class', '')
        gt_class = failure.get('gt_class', '')
        for pair in CONFUSED_PAIRS:
            if pred_class in pair and gt_class in pair:
                return 'confused_class'
    
    return 'other'


def visualize_failures(image: np.ndarray, predictions: Dict, 
                       ground_truth: Dict = None, title: str = "",
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Visualizza detection con GT (verde) e predizioni (rosso/blu).
    
    Predizioni corrette in blu, errori in rosso, GT in verde tratteggiato.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    ax.imshow(image)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Ground truth (verde tratteggiato)
    if ground_truth:
        gt_boxes = ground_truth.get('boxes', [])
        gt_labels = ground_truth.get('labels', [])
        
        for box, label in zip(gt_boxes, gt_labels):
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            cls_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f'cls_{label}'
            ax.text(box[0], box[1] - 5, f'GT: {cls_name}', 
                   color='green', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Predizioni
    pred_boxes = predictions.get('boxes', [])
    pred_labels = predictions.get('labels', [])
    pred_scores = predictions.get('scores', [])
    
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        # Determina colore: blu se match, rosso se errore
        color = 'red'
        if ground_truth:
            gt_boxes = ground_truth.get('boxes', [])
            gt_labels = ground_truth.get('labels', [])
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                if calculate_iou(box, gt_box) >= 0.5 and label == gt_label:
                    color = 'blue'
                    break
        
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        cls_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f'cls_{label}'
        ax.text(box[0], box[3] + 12, f'{cls_name}: {score:.2f}',
               color=color, fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    return fig


def visualize_failure_grid(failure_cases: List[Dict], n_cols: int = 3,
                           title: str = "") -> plt.Figure:
    """
    Visualizza griglia di failure cases.
    """
    n_cases = len(failure_cases)
    if n_cases == 0:
        return None
    
    n_rows = (n_cases + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=150)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, case in enumerate(failure_cases):
        ax = axes[i] if n_cases > 1 else axes[0]
        image = case.get('image')
        
        if image is not None:
            ax.imshow(image)
            
            # Disegna box
            box = case.get('box', [])
            if len(box) == 4:
                color = 'red' if case['type'] != 'false_negative' else 'orange'
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
        
        # Caption
        failure_type = case['type'].replace('_', ' ').title()
        if case['type'] == 'wrong_class':
            caption = f"{failure_type}\nGT: {case.get('gt_class', '?')} -> Pred: {case.get('pred_class', '?')}"
        elif case['type'] == 'false_negative':
            caption = f"{failure_type}\nClasse: {case.get('gt_class', '?')} ({case.get('size_category', '')})"
        else:
            caption = f"{failure_type}\nConf: {case.get('confidence', 0):.2f}"
        
        ax.set_title(caption, fontsize=9)
        ax.axis('off')
    
    # Nascondi assi vuoti
    for i in range(n_cases, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def load_coco_annotations(annotation_path: str) -> Dict[str, Dict]:
    """
    Carica annotazioni formato COCO.
    
    Returns:
        Dict filename -> {boxes, labels}
    """
    with open(annotation_path, 'r') as f:
        coco = json.load(f)
    
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    
    annotations = defaultdict(lambda: {'boxes': [], 'labels': []})
    
    for ann in coco['annotations']:
        filename = id_to_file.get(ann['image_id'], '')
        x, y, w, h = ann['bbox']
        annotations[filename]['boxes'].append([x, y, x + w, y + h])
        annotations[filename]['labels'].append(ann['category_id'])
    
    # Converti in numpy
    for fname in annotations:
        annotations[fname]['boxes'] = np.array(annotations[fname]['boxes'])
        annotations[fname]['labels'] = np.array(annotations[fname]['labels'])
    
    return dict(annotations)


def batch_inference(model: torch.nn.Module, image_paths: List[Path],
                    device: str, threshold: float = 0.5) -> Dict[str, Dict]:
    """
    Inference su batch di immagini con progress bar.
    """
    results = {}
    
    for img_path in tqdm(image_paths, desc="Inference"):
        try:
            tensor, img_rgb = preprocess_image(img_path, device)
            pred = run_inference(model, tensor, threshold)
            pred['image'] = img_rgb
            results[img_path.name] = pred
        except Exception as e:
            warnings.warn(f"Errore {img_path.name}: {e}")
    
    return results


def collect_failures(predictions: Dict[str, Dict], annotations: Dict[str, Dict],
                     iou_threshold: float = 0.5) -> Dict[str, List]:
    """
    Raccoglie tutti i failure dal dataset.
    
    Returns:
        Dict tipo_failure -> lista casi
    """
    all_failures = {
        'false_negative': [],
        'false_positive': [],
        'wrong_class': [],
        'poor_localization': []
    }
    
    for filename, pred in predictions.items():
        gt = annotations.get(filename, {'boxes': np.array([]), 'labels': np.array([])})
        failures = classify_failures(pred, gt, iou_threshold)
        
        for f in failures:
            f['filename'] = filename
            f['image'] = pred.get('image')
            all_failures[f['type']].append(f)
    
    return all_failures


def compute_class_metrics(predictions: Dict[str, Dict], annotations: Dict[str, Dict],
                          iou_threshold: float = 0.5) -> Dict[int, Dict]:
    """
    Calcola precision/recall per ogni classe.
    """
    metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for filename, pred in predictions.items():
        gt = annotations.get(filename, {'boxes': np.array([]), 'labels': np.array([])})
        
        gt_boxes = gt.get('boxes', np.array([]))
        gt_labels = gt.get('labels', np.array([]))
        if len(gt_boxes) == 0:
            gt_boxes = np.array([]).reshape(0, 4)
        
        matched = set()
        
        for pred_box, pred_label in zip(pred['boxes'], pred['labels']):
            is_tp = False
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if j in matched:
                    continue
                if calculate_iou(pred_box, gt_box) >= iou_threshold and pred_label == gt_label:
                    metrics[int(pred_label)]['tp'] += 1
                    matched.add(j)
                    is_tp = True
                    break
            if not is_tp:
                metrics[int(pred_label)]['fp'] += 1
        
        for j, gt_label in enumerate(gt_labels):
            if j not in matched:
                metrics[int(gt_label)]['fn'] += 1
    
    # Calcola metriche
    for cls_id in metrics:
        tp, fp, fn = metrics[cls_id]['tp'], metrics[cls_id]['fp'], metrics[cls_id]['fn']
        metrics[cls_id]['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics[cls_id]['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        p, r = metrics[cls_id]['precision'], metrics[cls_id]['recall']
        metrics[cls_id]['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    return dict(metrics)


def compute_size_stats(predictions: Dict[str, Dict], annotations: Dict[str, Dict],
                       iou_threshold: float = 0.5) -> Dict[str, Dict]:
    """
    Analizza detection rate per dimensione oggetto.
    """
    stats = {
        'small': {'detected': 0, 'total': 0},
        'medium': {'detected': 0, 'total': 0},
        'large': {'detected': 0, 'total': 0}
    }
    
    for filename, pred in predictions.items():
        gt = annotations.get(filename, {'boxes': np.array([]), 'labels': np.array([])})
        gt_boxes = gt.get('boxes', np.array([]))
        gt_labels = gt.get('labels', np.array([]))
        
        if len(gt_boxes) == 0:
            continue
        
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            _, _, area = get_box_dimensions(gt_box)
            size_cat = classify_size(area)
            stats[size_cat]['total'] += 1
            
            for pred_box, pred_label in zip(pred['boxes'], pred['labels']):
                if calculate_iou(pred_box, gt_box) >= iou_threshold and pred_label == gt_label:
                    stats[size_cat]['detected'] += 1
                    break
    
    for cat in stats:
        total = stats[cat]['total']
        stats[cat]['rate'] = stats[cat]['detected'] / total if total > 0 else 0
    
    return stats


def get_confusion_matrix(predictions: Dict[str, Dict], annotations: Dict[str, Dict],
                         classes: List[int] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Crea confusion matrix per le classi specificate.
    """
    if classes is None:
        # Usa le classi piu' comuni
        all_labels = []
        for pred in predictions.values():
            all_labels.extend(pred['labels'].tolist())
        for ann in annotations.values():
            all_labels.extend(ann['labels'].tolist())
        classes = sorted(set(all_labels))[:20]  # Top 20
    
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for filename, pred in predictions.items():
        gt = annotations.get(filename, {'boxes': np.array([]), 'labels': np.array([])})
        failures = classify_failures(pred, gt, iou_threshold=0.5)
        
        for f in failures:
            if f['type'] == 'wrong_class':
                gt_label = f.get('gt_label', -1)
                pred_label = f.get('pred_label', -1)
                if gt_label in class_to_idx and pred_label in class_to_idx:
                    matrix[class_to_idx[gt_label], class_to_idx[pred_label]] += 1
    
    class_names = [COCO_CLASSES[c] if c < len(COCO_CLASSES) else f'cls_{c}' for c in classes]
    return matrix, class_names
