"""
Rilevamento di Stelle Marine con Trasformata di Hough

Questo modulo fornisce utilità per rilevare stelle marine usando tre metodi:
1. Trasformata di Hough Generalizzata (GHT)
2. SIFT + RANSAC
3. Correlazione Incrociata Normalizzata (NCC) / Template Matching

Autore: Fabrizio Soppelsa
Corso: Visione Artificiale, UniPA, A.A. 2025/2026
"""

import cv2
import numpy as np
from pathlib import Path
from glob import glob
import random
import matplotlib.pyplot as plt

# Dimensione standard per normalizzare tutte le immagini
STANDARD_SIZE = (500, 500)  # larghezza, altezza


def show_results(results, method_name):
    """
    Visualizza i risultati del rilevamento in una griglia di 4 colonne.
    Colonne: Original | Canny | Accumulator/Match | Center
    
    Parametri:
        results: lista di dizionari con chiavi 'path', 'image', 'edges', 'accumulator', 'center', 'votes'
        method_name: nome del metodo (per il titolo)
    """
    n_images = len(results)
    fig, axes = plt.subplots(n_images, 4, figsize=(16, 4*n_images))
    
    # Assicurati che axes sia sempre 2D
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    # Per ogni immagine nei risultati
    for i in range(n_images):
        result = results[i]
        img_name = Path(result['path']).name
        center = result['center']
        votes = result.get('votes', 0)
        
        # Colonna 1: Immagine originale
        img_rgb = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f'{img_name}')
        axes[i, 0].axis('off')
        
        # Colonna 2: Bordi Canny
        if 'edges' in result and result['edges'] is not None:
            axes[i, 1].imshow(result['edges'], cmap='gray')
            axes[i, 1].set_title('Canny')
        else:
            axes[i, 1].imshow(np.zeros_like(result['image'][:,:,0]), cmap='gray')
            axes[i, 1].set_title('No bordi')
        axes[i, 1].axis('off')
        
        # Colonna 3: Accumulatore o Heatmap
        if 'accumulator' in result and result['accumulator'] is not None:
            axes[i, 2].imshow(result['accumulator'], cmap='hot')
            if center is not None:
                axes[i, 2].plot(center[0], center[1], 'b*', markersize=20)
            axes[i, 2].set_title(f'Accumulator')
        elif 'heatmap' in result and result['heatmap'] is not None:
            axes[i, 2].imshow(result['heatmap'], cmap='hot')
            if center is not None:
                axes[i, 2].plot(center[0], center[1], 'b*', markersize=15)
            axes[i, 2].set_title('Match')
        else:
            axes[i, 2].imshow(np.zeros_like(result['image'][:,:,0]), cmap='gray')
            axes[i, 2].set_title('No match')
        axes[i, 2].axis('off')
        
        # Colonna 4: Centro rilevato
        img_result = result['image'].copy()
        if center is not None:
            cv2.circle(img_result, center, 20, (0, 255, 0), 3)
            cv2.drawMarker(img_result, center, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            axes[i, 3].set_title(f'Centro imm: {center}')
        else:
            axes[i, 3].set_title('Centro non rilevato')
        img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        axes[i, 3].imshow(img_result_rgb)
        axes[i, 3].axis('off')
    
    fig.suptitle(f'{method_name} - Risultati Rilevamento', fontsize=16, y=1.0)
    plt.tight_layout()
    plt.show()


def detect_starfish_ght(image_path, template_edges, template_center, standard_size=STANDARD_SIZE):
    """
    Rileva stella marina usando Generalized Hough Transform (OpenCV).
    
    Parametri:
        image_path: percorso dell'immagine
        template_edges: bordi del template
        template_center: centro del template (x, y)
        standard_size: dimensione di normalizzazione (larghezza, altezza)
    
    Ritorna:
        dizionario con 'path', 'image', 'edges', 'accumulator', 'center', 'votes'
    """
    # 1. Carica immagine target
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Normalizza dimensioni
    img = cv2.resize(img, standard_size)
    
    # 2. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    target_edges = cv2.Canny(gray_blurred, 50, 150)
    
    # 3. Crea detector OpenCV Generalized Hough Ballard
    ght = cv2.createGeneralizedHoughBallard()
    
    # 4. Imposta il template
    ght.setTemplate(template_edges, template_center)
    
    # 5. Parametri del detector
    ght.setMinDist(50)
    ght.setDp(2)
    ght.setCannyLowThresh(50)
    ght.setCannyHighThresh(150)
    ght.setLevels(360)
    ght.setVotesThreshold(15)
    
    # 6. Rileva usando GHT
    try:
        positions = ght.detect(target_edges)
    except:
        positions = None
    
    center = None
    votes = 0
    
    # 7. Estrai risultati
    if positions is not None:
        if len(positions) > 0:
            if positions[0] is not None:
                if len(positions[0]) > 0:
                    pos = positions[0][0]
                    pos_flat = np.array(pos).flatten()
                    if len(pos_flat) >= 2:
                        center = (int(pos_flat[0]), int(pos_flat[1]))
                        votes = len(positions[0]) * 50
    
    # 8. Crea accumulator simulato (per visualizzazione)
    accumulator = None
    if center is not None:
        h = img.shape[0]
        w = img.shape[1]
        accumulator = np.zeros((h, w), dtype=np.float32)
        # Crea un picco gaussiano al centro rilevato
        y, x = np.ogrid[:h, :w]
        distanza_quadrata = (x - center[0])**2 + (y - center[1])**2
        mask = distanza_quadrata <= (50**2)
        accumulator[mask] = votes / 100.0
        if accumulator.max() > 0:
            accumulator = accumulator / accumulator.max()
    
    return {
        'path': image_path,
        'image': img,
        'edges': target_edges,
        'accumulator': accumulator,
        'center': center,
        'votes': votes
    }


def detect_starfish_sift(image_path, template_gray, standard_size=STANDARD_SIZE, min_inliers=4):
    """
    Rileva stella marina usando SIFT + RANSAC.
    
    Parametri:
        image_path: percorso dell'immagine
        template_gray: template in grayscale
        standard_size: dimensione di normalizzazione (larghezza, altezza)
        min_inliers: numero minimo di inliers per validare il rilevamento (default 4)
    
    Ritorna:
        dizionario con 'path', 'image', 'edges', 'accumulator', 'center', 'votes'
    """
    # 1. Carica immagine target
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Normalizza dimensioni
    img = cv2.resize(img, standard_size)
    
    # 2. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(gray_blurred, 50, 150)
    
    # 3. Inizializza SIFT detector con parametri molto sensibili
    sift = cv2.SIFT_create(
        nfeatures=0,              # Nessun limite sul numero di features
        contrastThreshold=0.02,   # Molto basso = molte più features (default 0.04)
        edgeThreshold=15,         # Molto alto = più features su bordi (default 10)
        sigma=1.6                 # Smoothing iniziale
    )
    
    # 4. Trova keypoints e descriptors
    kp1, des1 = sift.detectAndCompute(template_gray, None)
    kp2, des2 = sift.detectAndCompute(gray, None)
    
    # Conta keypoints trovati
    if kp1:
        n_kp1 = len(kp1)
    else:
        n_kp1 = 0
    if kp2:
        n_kp2 = len(kp2)
    else:
        n_kp2 = 0
    #print(f"  Template che provo: {n_kp1} keypoints, Target: {n_kp2} keypoints")
    
    # Controlla se abbiamo abbastanza keypoints
    if des1 is None or des2 is None:
        return {
            'path': image_path,
            'image': img,
            'edges': edges,
            'accumulator': None,
            'center': None,
            'votes': 0,
            'n_matches': 0,
            'n_inliers': 0
        }
    if len(kp1) < 4 or len(kp2) < 4:
        return {
            'path': image_path,
            'image': img,
            'edges': edges,
            'accumulator': None,
            'center': None,
            'votes': 0,
            'n_matches': 0,
            'n_inliers': 0
        }
    
    # 5. Match features con BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # 6. Applica Lowe's ratio test (molto permissivo: 0.85)
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m = match[0]
            n = match[1]
            if m.distance < 0.85 * n.distance:
                good_matches.append(m)
        elif len(match) == 1:
            # Se c'è solo un match, accettalo comunque
            good_matches.append(match[0])
    
    n_matches = len(good_matches)
#    print(f"  Match trovati: {n_matches}")
    
    if n_matches < 4:
        return {
            'path': image_path,
            'image': img,
            'edges': edges,
            'accumulator': None,
            'center': None,
            'votes': n_matches,
            'n_matches': n_matches,
            'n_inliers': 0
        }
    
    # 7. Estrai punti corrispondenti
    src_pts_list = []
    for m in good_matches:
        pt = kp1[m.queryIdx].pt
        src_pts_list.append(pt)
    src_pts = np.float32(src_pts_list).reshape(-1, 1, 2)
    
    dst_pts_list = []
    for m in good_matches:
        pt = kp2[m.trainIdx].pt
        dst_pts_list.append(pt)
    dst_pts = np.float32(dst_pts_list).reshape(-1, 1, 2)
    
    # 8. Trova omografia con RANSAC (ransacReprojThreshold ok permissivo)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
    
    n_inliers = 0
    center = None
    accumulator = None
    
    if H is not None:
        mask = mask.ravel()
        n_inliers = int(np.sum(mask))
        
        #print(f"  Inlier: {n_inliers}")
        
        if n_inliers >= min_inliers:
            # 9. Calcola il centro del template trasformato
            h = template_gray.shape[0]
            w = template_gray.shape[1]
            template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            target_corners = cv2.perspectiveTransform(template_corners, H)
            
            # Centro = media dei 4 angoli trasformati
            sum_x = 0
            sum_y = 0
            for i in range(4):
                sum_x = sum_x + target_corners[i, 0, 0]
                sum_y = sum_y + target_corners[i, 0, 1]
            center_x = int(sum_x / 4)
            center_y = int(sum_y / 4)
            center = (center_x, center_y)
            
            # Crea accumulator simulato (heatmap dei match inliers)
            h_img = img.shape[0]
            w_img = img.shape[1]
            accumulator = np.zeros((h_img, w_img), dtype=np.float32)
            for i in range(len(good_matches)):
                if mask[i]:
                    m = good_matches[i]
                    pt = kp2[m.trainIdx].pt
                    x = int(pt[0])
                    y = int(pt[1])
                    if 0 <= y < h_img and 0 <= x < w_img:
                        cv2.circle(accumulator, (x, y), 10, 1.0, -1)
            if accumulator.max() > 0:
                accumulator = accumulator / accumulator.max()
    
    return {
        'path': image_path,
        'image': img,
        'edges': edges,
        'accumulator': accumulator,
        'center': center,
        'votes': n_inliers,
        'n_matches': n_matches,
        'n_inliers': n_inliers
    }


def detect_starfish_ncc(image_path, template_gray, 
                        template_edges, standard_size=STANDARD_SIZE, threshold=0.1):
    """
    Rileva stella marina usando Normalized Cross-Correlation (Template Matching).
    Usa principalmente edge matching che è più robusto.
    Prova diversi metodi di matching e sceglie il migliore.
    
    Parametri:
        image_path: percorso dell'immagine
        template_gray: template in grayscale
        template_edges: bordi del template (Canny)
        standard_size: dimensione di normalizzazione (larghezza, altezza)
        threshold: soglia minima di matching (default 0.1)
    
    Ritorna:
        dizionario con 'path', 'image', 'edges', 'heatmap', 'center', 'votes'
    """
    # 1. Carica immagine target
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Normalizza dimensioni
    img = cv2.resize(img, standard_size)
    
    # 2. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(gray_blurred, 50, 150)
    
    # 3. Prova molte scale diverse
    scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    # Prova diversi metodi di matching
    methods = [
        ('CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
        ('CCORR_NORMED', cv2.TM_CCORR_NORMED),
    ]
    
    best_overall_val = -1
    best_center = None
    best_heatmap = None
    best_scale = 1.0
    best_method = None
    all_results = []
    
    template_h = template_edges.shape[0]
    template_w = template_edges.shape[1]
    
    for method_item in methods:
        method_name = method_item[0]
        method = method_item[1]
        
        for scale in scales:
            # Ridimensiona template alla scala corrente
            new_w = int(template_w * scale)
            new_h = int(template_h * scale)
            
            # Salta scale troppo grandi o piccole
            if new_w >= edges.shape[1] - 5:
                continue
            if new_h >= edges.shape[0] - 5:
                continue
            if new_w < 30:
                continue
            if new_h < 30:
                continue
            
            # Ridimensiona template edges
            template_edges_resized = cv2.resize(template_edges, (new_w, new_h))
            
            # 4. Match su EDGES
            result_edges = cv2.matchTemplate(edges, template_edges_resized, method)
            
            if method == cv2.TM_SQDIFF_NORMED:
                # Per SQDIFF, il minimo è il migliore
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_edges)
                match_val = 1.0 - min_val  # Inverti per avere un valore "alto = buono"
                match_loc = min_loc
            else:
                # Per CCOEFF e CCORR, il massimo è il migliore
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_edges)
                match_val = max_val
                match_loc = max_loc
            
            all_results.append((method_name, scale, match_val))
            
            # Aggiorna se questo è il miglior match finora
            if match_val > best_overall_val:
                best_overall_val = match_val
                best_scale = scale
                best_method = method_name
                h = template_edges_resized.shape[0]
                w = template_edges_resized.shape[1]
                best_center = (match_loc[0] + w // 2, match_loc[1] + h // 2)
                
                # Normalizza heatmap
                heatmap_normalized = result_edges.copy()
                min_v = heatmap_normalized.min()
                max_v = heatmap_normalized.max()
                if max_v > min_v:
                    heatmap_normalized = (heatmap_normalized - min_v) / (max_v - min_v)
                best_heatmap = heatmap_normalized
    
    # Ordina i risultati per valore di match (decrescente)
    for i in range(len(all_results)):
        for j in range(i + 1, len(all_results)):
            if all_results[j][2] > all_results[i][2]:
                # Scambia
                temp = all_results[i]
                all_results[i] = all_results[j]
                all_results[j] = temp
    
    # Mostra i top 5 match per debug
    top5_list = []
    for i in range(min(5, len(all_results))):
        m = all_results[i][0]
        s = all_results[i][1]
        v = all_results[i][2]
        top5_list.append((m, f'{s:.1f}', f'{v:.3f}'))
    #print(f"  Top 5: {top5_list}")
    #print(f"  Best: {best_overall_val:.4f} ({best_method} @ {best_scale:.2f}x) [soglia: {threshold}]")
    
    center = None
    votes = best_overall_val
    
    # 5. Valida il match con la soglia (molto permissivo!)
    if best_overall_val >= threshold:
        center = best_center
    
    # 6. Normalizza heatmap per visualizzazione
    heatmap = best_heatmap
    
    return {
        'path': image_path,
        'image': img,
        'edges': edges,
        'heatmap': heatmap,
        'center': center,
        'votes': votes
    }


def benchmark(data_dir, template_name, n_images=30, distance_threshold=50, random_seed=42, show_n_examples=5):
    """
    Esegue benchmark completo dei tre metodi con ground truth.
    
    Parametri:
        data_dir: directory con le immagini
        template_name: nome del file template
        n_images: numero di immagini da testare (default 30)
        distance_threshold: soglia distanza per TP (default 50px)
        random_seed: seed per riproducibilità (default 42)
        show_n_examples: numero di esempi da visualizzare per metodo (default 5)
    
    Ritorna:
        dict con risultati completi per ogni metodo
    """
    import random
    import json
    
    data_dir = Path(data_dir)
    gt_file = data_dir / "ground_truth.json"
    
    print("BENCHMARK STELLE MARINE")
    
    # Verifica e carica ground truth
    if not gt_file.exists():
        print(f"ERROR: File ground truth non trovato: {gt_file}")
        return None
    
    ground_truth = load_ground_truth(gt_file)
    valid_gt = {k: v for k, v in ground_truth.items() if v.get('present', False)}
    print(f"Ground truth: {len(valid_gt)} immagini")
    
    # Trova e carica template
    template_path = None
    for img_file in data_dir.glob("*.jpg"):
        if template_name in img_file.name:
            template_path = img_file
            break
    
    if template_path is None:
        print(f"ERROR: Template non trovato: {template_name}")
        return None
    
    template = cv2.imread(str(template_path))
    if template is None:
        print(f"ERROR: Impossibile caricare template")
        return None
    
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    print(f"Template: {template_path.name}")
    
    # Seleziona immagini per test
    available_images = [str(f) for f in data_dir.glob("*.jpg")]
    test_candidates = [img for img in available_images if template_name not in img]
    
    random.seed(random_seed)
    if len(test_candidates) >= n_images:
        selected_images = random.sample(test_candidates, n_images)
    else:
        selected_images = test_candidates
    
    print(f"Test: {len(selected_images)} immagini, soglia: {distance_threshold}px")
    
    results = compare_methods_with_ground_truth(
        image_paths=selected_images,
        template_gray=template_gray,
        ground_truth=ground_truth,
        distance_threshold=distance_threshold
    )
    
    # Riepilogo - tabella
    print()
    print(f"{'Metodo':<10} {'Precision':<12} {'Recall':<12} {'Tempo/Img':<12} {'Dist.Media':<12}")
    
    for method_name in ['ght', 'sift', 'ncc']:
        eval_data = results[method_name]['evaluation']
        precision = eval_data['precision']
        recall = eval_data['recall']
        avg_time = results[method_name]['avg_time']
        avg_dist = eval_data['avg_distance']
        
        method_label = method_name.upper()
        print(f"{method_label:<10} {precision:<12.2%} {recall:<12.2%} {avg_time:<12.3f}s", end='')
        if avg_dist is not None:
            print(f" {avg_dist:<12.1f} px")
        else:
            print(f" {'N/A':<12}")
    
    # Plotta grafici
    plot_comparison_with_ground_truth(results)
    
    # Mostra esempi stelle per ogni metodo
    if show_n_examples > 0:
        for method_name in ['ght', 'sift', 'ncc']:
            method_results = results[method_name]['results']
            show_results(method_results[:show_n_examples], f"{method_name.upper()}")
    
    return results


def load_ground_truth(gt_file):
    """
    Carica il ground truth da file JSON.
    
    Parametri:
        gt_file: path del file JSON con ground truth
    
    Ritorna:
        dict: {filename: {'center': [x, y], 'present': True/False, ...}}
    """
    import json
    with open(gt_file, 'r') as f:
        return json.load(f)


def evaluate_with_ground_truth(results, ground_truth, distance_threshold=50):
    """
    Valuta i risultati del rilevamento confrontandoli con ground truth.
    
    Parametri:
        results: lista di risultati da detect_starfish_*
        ground_truth: dict da load_ground_truth()
        distance_threshold: distanza massima in pixel per considerare una detection corretta
    
    Ritorna:
        dict con metriche: {
            'precision': 0.0-1.0,
            'recall': 0.0-1.0,
            'f1_score': 0.0-1.0,
            'true_positives': int,
            'false_positives': int,
            'false_negatives': int,
            'avg_distance': float,  # distanza media per TP
            'details': lista di dict per ogni immagine
        }
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    distances = []
    details = []
    
    for result in results:
        img_name = Path(result['path']).name
        gt_data = ground_truth.get(img_name, {})
        
        detected_center = result['center']
        
        detail = {
            'image': img_name,
            'ground_truth_present': gt_data.get('present', False),
            'detected': detected_center is not None,
            'ground_truth_center': gt_data.get('center'),
            'detected_center': detected_center,
            'status': None,
            'distance': None
        }
        
        if gt_data.get('present', False):
            # Stella marina presente nella ground truth
            gt_center = gt_data['center']
            
            if detected_center is not None:
                # Calcola distanza euclidea
                dist = np.sqrt((detected_center[0] - gt_center[0])**2 + 
                              (detected_center[1] - gt_center[1])**2)
                detail['distance'] = float(dist)
                
                if dist <= distance_threshold:
                    true_positives += 1
                    detail['status'] = 'TP'  # True Positive
                    distances.append(dist)
                else:
                    false_positives += 1
                    detail['status'] = 'FP_wrong_location'  # Rilevato ma sbagliato
            else:
                false_negatives += 1
                detail['status'] = 'FN'  # False Negative (mancato rilevamento)
        else:
            # Nessuna stella nella ground truth
            if detected_center is not None:
                false_positives += 1
                detail['status'] = 'FP'  # False Positive
            else:
                # True Negative (correttamente non rilevato)
                detail['status'] = 'TN'
        
        details.append(detail)
    
    # Ritorna le metriche
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_distance = np.mean(distances) if distances else None
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'avg_distance': avg_distance,
        'details': details
    }


def compare_methods_with_ground_truth(image_paths, template_gray, ground_truth, 
                                      distance_threshold=50, standard_size=STANDARD_SIZE):
    """
    Confronta tutti e tre i metodi usando ground truth.
    
    Parametri:
        image_paths: lista di percorsi immagini da testare
        template_gray: immagine template in scala di grigi
        ground_truth: dict da load_ground_truth()
        distance_threshold: soglia distanza per TP (default 50px)
        standard_size: dimensione normalizzata
    
    Ritorna:
        dict con risultati per metodo: {
            'ght': {'results': [...], 'evaluation': {...}, 'avg_time': float},
            'sift': {'results': [...], 'evaluation': {...}, 'avg_time': float},
            'ncc': {'results': [...], 'evaluation': {...}, 'avg_time': float}
        }
    """
    import time
    
    # Prepara template edges e center per GHT e NCC
    gray_blurred = cv2.GaussianBlur(template_gray, (5, 5), 1.4)
    template_edges = cv2.Canny(gray_blurred, 50, 150)
    edge_points = np.column_stack(np.where(template_edges > 0))
    center_y, center_x = np.mean(edge_points, axis=0).astype(int)
    template_center = (center_x, center_y)
    
    all_results = {}
    n_images = len(image_paths)
    
    # Test GHT
    print(f"\nTesto GHT...")
    start_time = time.time()
    ght_results = []
    for img_path in image_paths:
        result = detect_starfish_ght(img_path, template_edges, template_center, standard_size)
        ght_results.append(result)
    ght_time = time.time() - start_time
    
    evaluation = evaluate_with_ground_truth(ght_results, ground_truth, distance_threshold)
    all_results['ght'] = {'results': ght_results, 'evaluation': evaluation, 'avg_time': ght_time / n_images}
    
    # Test SIFT
    print(f"\nTesto SIFT...")
    start_time = time.time()
    sift_results = []
    for img_path in image_paths:
        result = detect_starfish_sift(img_path, template_gray, standard_size)
        sift_results.append(result)
    sift_time = time.time() - start_time
    
    evaluation = evaluate_with_ground_truth(sift_results, ground_truth, distance_threshold)
    all_results['sift'] = {'results': sift_results, 'evaluation': evaluation, 'avg_time': sift_time / n_images}
    
    # Test NCC
    print(f"\nTesto NCC...")
    start_time = time.time()
    ncc_results = []
    for img_path in image_paths:
        result = detect_starfish_ncc(img_path, template_gray, template_edges, standard_size)
        ncc_results.append(result)
    ncc_time = time.time() - start_time
    
    evaluation = evaluate_with_ground_truth(ncc_results, ground_truth, distance_threshold)
    all_results['ncc'] = {'results': ncc_results, 'evaluation': evaluation, 'avg_time': ncc_time / n_images}
    
    return all_results


def plot_comparison_with_ground_truth(comparison_results):
    """
    Visualizza confronto tra metodi con metriche da ground truth.
    
    Parametri:
        comparison_results: output da compare_methods_with_ground_truth()
    """
    methods = ['GHT', 'SIFT', 'NCC']
    colors = ['red', 'blue', 'green']
    
    # Estrai metriche
    precisions = [comparison_results[m.lower()]['evaluation']['precision'] for m in methods]
    recalls = [comparison_results[m.lower()]['evaluation']['recall'] for m in methods]
    avg_times = [comparison_results[m.lower()]['avg_time'] * 1000 for m in methods]  # Convert to ms
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Confronto Metodi con Ground Truth', fontsize=16, fontweight='bold')
    
    # Precision
    axes[0].bar(methods, precisions, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title('Precision', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    for i, (m, p) in enumerate(zip(methods, precisions)):
        axes[0].text(i, p + 0.02, f'{p:.1%}', ha='center', fontsize=11, fontweight='bold')
    
    # Recall
    axes[1].bar(methods, recalls, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('Recall', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    for i, (m, r) in enumerate(zip(methods, recalls)):
        axes[1].text(i, r + 0.02, f'{r:.1%}', ha='center', fontsize=11, fontweight='bold')
    
    # Average Time per Image
    axes[2].bar(methods, avg_times, color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Tempo (ms)', fontsize=12)
    max_time = max(avg_times)
    axes[2].set_ylim(0, max_time * 1.15)
    axes[2].set_title('Tempo Medio per Immagine', fontsize=14, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    for i, (m, t) in enumerate(zip(methods, avg_times)):
        axes[2].text(i, t + max_time * 0.02, f'{t:.1f}ms', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Confusion Matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Conteggi per metodo', fontsize=16, fontweight='bold')
    
    for i, method in enumerate(methods):
        eval_data = comparison_results[method.lower()]['evaluation']
        
        # Matrice confusione
        tp = eval_data['true_positives']
        fp = eval_data['false_positives']
        fn = eval_data['false_negatives']
        
        categories = ['TP', 'FP', 'FN']
        values = [tp, fp, fn]
        bar_colors = ['green', 'red', 'orange']
        
        axes[i].bar(categories, values, color=bar_colors, alpha=0.7, edgecolor='black')
        axes[i].set_ylabel('Conteggio', fontsize=12)
        axes[i].set_title(f'{method}', fontsize=14, fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3)
        
        for j, (cat, val) in enumerate(zip(categories, values)):
            axes[i].text(j, val + 0.1, str(val), ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
