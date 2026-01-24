"""
Benchmark di MobileNet, GhostNet, SqueezeNet
Confronto su parametri, FLOPs, latenza, accuratezza
UniPA - Visione Artificiale A.A. 2025/2026
"""

from torchvision import models, transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import os
import random
import time
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
import timm
import torch
import torch.nn as nn
from thop import profile

# Conta parametri modello
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Calcola flops
# https://github.com/ultralytics/thop
def count_flops(model, input_size=(1, 3, 224, 224)):
    try:
        input_tensor = torch.randn(input_size).to(next(model.parameters()).device)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops
    except Exception as e:
        print(f"Eccezione calcolo FLOPs: {e}")
        return 0

# Misura latenza media del modello
def measure_latency(model, input_size=(1, 3, 224, 224), num_runs=100, warmup=10):
    device = next(model.parameters()).device
    model.eval()
    
    # Crea input dummy
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Sincronizza GPU se disponibile
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Misura
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # converti in ms
    
    return np.mean(times), np.std(times)


def compute_accuracy(model, batch_size=16, device='cuda', return_errors=False):
    """
    Calcola accuratezza del modello su dataset semplice con categorie note
    Misura Top-1 e Top-5 accuracy
    
    Args:
        model: modello PyTorch
        batch_size: batch size per inference
        device: device PyTorch
        return_errors: se True, restituisce anche lista errori
    
    Returns:
        dict con metriche: top1_acc, top5_acc, num_images, [errors]
    """
    images_dir = "data/mobilenet/images"
    
    # Categorie e mapping ImageNet class index
    # Questi sono gli indici ImageNet per le categorie
    CATEGORY_TO_IMAGENET = {
        'dog': list(range(151, 276)),  # tipi di dogs in ImageNet (151-275)
        'cat': list(range(281, 286)),  # tipi di cats (281-285)
        'horse': [339, 340],  # horse, zebra
        'bird': list(range(7, 24)) + list(range(80, 101)),  # vari uccelli
        'car': [436, 511, 609, 627, 656, 661, 751, 817],
        'truck': [555, 569, 717, 734, 864, 867],
        'airplane': [404, 895],
        'boat': [472, 554, 625, 814, 914],  # canoe, speedboat, lifeboat, etc
        'bicycle': [444, 671],
        'chair': [423, 559, 765],
    }
    print(f"  Valutazione accuratezza su dataset semplice...")
    
    # Transform standard per ImageNet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    model.to(device).eval()
    
    # Valuta per categoria
    top1_correct = 0
    top5_correct = 0
    total_evaluated = 0
    errors = [] if return_errors else None
    
    with torch.no_grad():
        for category, imagenet_indices in CATEGORY_TO_IMAGENET.items():
            category_dir = Path(images_dir) / category
            if not category_dir.exists():
                print(f"  {category}: directory non trovata")
                continue
            
            image_files = list(category_dir.glob('*.jpg')) + list(category_dir.glob('*.png'))
            
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i+batch_size]
                batch_images = []
                
                for img_file in batch_files:
                    try:
                        img = Image.open(img_file).convert('RGB')
                        img_tensor = transform(img)
                        batch_images.append(img_tensor)
                    except Exception:
                        continue
                
                if len(batch_images) == 0:
                    continue
                
                # Stack batch
                batch_tensor = torch.stack(batch_images).to(device)
                
                # Inference
                outputs = model(batch_tensor)
                
                # Get top-5 predictions
                _, top5_indices = outputs.topk(5, dim=1)
                
                # Check accuracy
                for j in range(len(batch_images)):
                    top1_pred = top5_indices[j][0].item()
                    top5_preds = top5_indices[j].cpu().numpy()
                    
                    # Check if prediction matches category
                    is_top1_correct = top1_pred in imagenet_indices
                    is_top5_correct = any(pred in imagenet_indices for pred in top5_preds)
                    
                    if is_top1_correct:
                        top1_correct += 1
                    
                    if is_top5_correct:
                        top5_correct += 1
                    
                    # Salva errore se richiesto
                    if return_errors and not is_top1_correct:
                        errors.append({
                            'image_path': batch_files[j],
                            'true_category': category,
                            'pred_idx': top1_pred,
                        })
                    
                    total_evaluated += 1
    
    # Calcola accuratezza
    top1_acc = (top1_correct / total_evaluated * 100) if total_evaluated > 0 else 0.0
    top5_acc = (top5_correct / total_evaluated * 100) if total_evaluated > 0 else 0.0
    
    print(f"  Immagini valutate: {total_evaluated}")
    print(f"    Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"    Top-5 Accuracy: {top5_acc:.2f}%")
    
    result = {
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "num_images": total_evaluated
    }
    
    if return_errors:
        result["errors"] = errors
    
    return result

# Benchmark di un modello PyTorch
def benchmark_model(model, model_name, input_size=(1, 3, 224, 224), device='cpu'):
    print(f"\nBenchmark di {model_name}...")
    
    model.to(device).eval()
    
    # Parametri
    params = count_parameters(model)
    print(f"  Parametri: {params:,} ({params/1e6:.2f}M)")
    
    # FLOPs
    flops = count_flops(model, input_size)
    if flops > 0:
        print(f"  FLOPs: {flops:,} ({flops/1e9:.2f}G)")
    
    # Latenza
    latency_mean, latency_std = measure_latency(model, input_size, num_runs=50, warmup=5)
    fps = 1000.0 / latency_mean if latency_mean > 0 else 0
    print(f"  Latenza: {latency_mean:.2f} +- {latency_std:.2f} ms")
    print(f"  FPS: {fps:.1f}")
    
    # Accuratezza
    accuracy_results = compute_accuracy(model, batch_size=16, device=device)
    
    return {
        "model": model_name,
        "params": params,
        "params_M": params / 1e6,
        "flops": flops,
        "flops_G": flops / 1e9 if flops > 0 else 0,
        "latency_ms": latency_mean,
        "latency_std_ms": latency_std,
        "fps": fps,
        "top1_acc": accuracy_results['top1_acc'],
        "top5_acc": accuracy_results['top5_acc'],
        "num_images_evaluated": accuracy_results['num_images'],
        "device": str(device),
    }

# Carica il modello con i pesi da torchvision o timm
def load_model(model_name, pretrained=True, device='cpu'):
    model_name_lower = model_name.lower()
    
    # MobileNetV1
    # https://huggingface.co/timm/mobilenetv1_100.ra_in1k
    if 'mobilenet_v1' in model_name_lower:
        model = timm.create_model('mobilenetv1_100', pretrained=pretrained)
    # MobileNetv2
    # https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html#torchvision.models.mobilenet_v2
    elif 'mobilenet_v2' in model_name_lower:
        model = models.mobilenet_v2(weights='IMAGENET1K_V2' if pretrained else None)
    # MobilenetV3_large
    # https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html#torchvision.models.mobilenet_v3_large
    elif 'mobilenet_v3_large' in model_name_lower:
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V2' if pretrained else None)
    # MobilenetV3_small
    # https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small
    elif 'mobilenet_v3_small' in model_name_lower:
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
    # MobilenetV4
    # https://huggingface.co/blog/rwightman/mobilenetv4
    elif 'mobilenet_v4' in model_name_lower:
        model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=pretrained)
    # SqueezeNet 1.0
    # https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.squeezenet1_0.html#torchvision.models.squeezenet1_0
    elif 'squeezenet1_0' in model_name_lower:
        model = models.squeezenet1_0(weights='IMAGENET1K_V1' if pretrained else None)
    # SqueezeNet 1.1
    # https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.squeezenet1_1.html#torchvision.models.squeezenet1_1
    elif 'squeezenet1_1' in model_name_lower:
        model = models.squeezenet1_1(weights='IMAGENET1K_V1' if pretrained else None)
    # GhostNet 
    # https://huggingface.co/timm/ghostnet_100.in1k
    elif 'ghostnet' in model_name_lower:
        model = timm.create_model('ghostnet_100', pretrained=pretrained)
    else:
        raise ValueError(f"Qualche errore, modello non supportato: {model_name}")

    return model.to(device)


def visualize_results(all_results):
    # Converti in DataFrame
    df = pd.DataFrame(all_results)
    models = df['model'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Confronto modelli CNN Efficienti', fontsize=16, fontweight='bold')
    
    # 1. Parametri
    axes[0, 0].bar(models, df['params_M'], color='steelblue', alpha=0.7)
    axes[0, 0].set_ylabel('Parametri (M)', fontsize=11)
    axes[0, 0].set_title('Parametri del Modello', fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. FLOPs
    axes[0, 1].bar(models, df['flops_G'], color='coral', alpha=0.7)
    axes[0, 1].set_ylabel('FLOPs (G)', fontsize=11)
    axes[0, 1].set_title('Complessità Computazionale', fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Latency
    axes[1, 0].bar(models, df['latency_ms'], color='mediumseagreen', alpha=0.7)
    axes[1, 0].set_ylabel('Latenza (ms)', fontsize=11)
    axes[1, 0].set_title('Latenza di Inferenza', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. FPS
    axes[1, 1].bar(models, df['fps'], color='mediumpurple', alpha=0.7)
    axes[1, 1].set_ylabel('FPS', fontsize=11)
    axes[1, 1].set_title('Frame Per Second', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # 5. Top-1 Accuracy
    if 'top1_acc' in df.columns:
        axes[0, 2].bar(models, df['top1_acc'], color='lightcoral', alpha=0.7)
        axes[0, 2].set_ylabel('Top-1 (%)', fontsize=11)
        axes[0, 2].set_title('Top-1', fontweight='bold')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(axis='y', alpha=0.3)
    
    # 6. Top-5 Accuracy
    if 'top5_acc' in df.columns:
        axes[1, 2].bar(models, df['top5_acc'], color='lightgreen', alpha=0.7)
        axes[1, 2].set_ylabel('Top-5 (%)', fontsize=11)
        axes[1, 2].set_title('Top-5', fontweight='bold')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Stampa tabella riassuntiva
    print("CONFRONTO MODELLI".center(80))
    cols_to_show = ['model', 'params_M', 'flops_G', 'latency_ms', 'fps']
    if 'top1_acc' in df.columns:
        cols_to_show.extend(['top1_acc', 'top5_acc'])
    print(tabulate(df[cols_to_show], headers='keys', tablefmt='grid', showindex=False, floatfmt='.2f'))


def mobilenetv3_pipeline(image, device='cpu', model=None, conf_threshold=0.5):
    """
    Pipeline completa per object detection utilizzando MobileNetV3-Large + SSDLite.
    
    Usa il modello pre-addestrato di torchvision che combina:
    - Backbone: MobileNetV3-Large 
    - Detection head: SSDLite (versione leggera di SSD)
    - Training: COCO dataset (91 classi)
    
    Args:
        image (PIL.Image o str): Immagine PIL o path al file immagine
        device (str, opzionale): Device PyTorch ('cpu' o 'cuda'). Se None, rileva automaticamente.
        model (torch.nn.Module, opzionale): Modello pre-caricato. Se None, carica nuovo modello.
        conf_threshold (float): Soglia di confidenza minima per mantenere le detection (0-1)
    
    Returns:
        dict: Dizionario contenente:
            - 'boxes' (ndarray): Array (N, 4) con coordinate bounding box [x1, y1, x2, y2]
            - 'scores' (ndarray): Array (N,) con confidence score per ogni detection
            - 'labels' (ndarray): Array (N,) con label della classe predetta (1-91 COCO)
            - 'num_detections' (int): Numero totale di detection sopra la soglia
    """
    # Pre-processing: trasformazione in tensor, il modello fa resize internamente
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converti da PIL a Tensor [0, 1]
    ])
    
    # Carica immagine da file se è stato passato un path
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    
    # Applica trasformazioni (il modello fa resize e normalizzazione internamente)
    img_tensor = transform(image).to(device)
    
    # Carica modello solo se non fornito (per evitare reload ripetuti in video)
    if model is None:
        model = ssdlite320_mobilenet_v3_large(weights='COCO_V1')
        model.to(device).eval()  # Modalità evaluation (disabilita dropout, batch norm, ecc.)
    
    # Inferenza senza calcolo dei gradienti (più veloce e usa meno memoria)
    with torch.no_grad():
        predictions = model([img_tensor])[0]  # Il modello vuole una lista di tensori
    
    # Post-processing delle predizioni: filtra per confidence threshold
    boxes = predictions['boxes'].cpu().numpy()  # Shape: (N, 4) - coordinate [x1, y1, x2, y2]
    scores = predictions['scores'].cpu().numpy()  # Shape: (N,) - confidence scores
    labels = predictions['labels'].cpu().numpy()  # Shape: (N,) - class labels (1-91)
    
    # Filtraggio: mantieni solo detection con confidence > threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Restituisci risultati
    return {
        'boxes': boxes,  # Coordinate dei box filtrati [x1, y1, x2, y2]
        'scores': scores,  # Confidence score dei box filtrati
        'labels': labels,  # Label delle classi predette (1-91 COCO)
        'num_detections': len(boxes)  # Numero totale di detection valide
    }

