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


def select_random_images(image_dir, seed=42, n=6, exclude_template=None):
    """
    Seleziona casualmente n immagini dalla directory.
    
    Parametri:
        image_dir: percorso della directory con le immagini
        seed: seed per la generazione casuale (default 42)
        n: numero di immagini da selezionare (default 6)
        exclude_template: path dell'immagine template da escludere
    
    Ritorna:
        Lista di percorsi delle immagini selezionate
    """
    # Trova tutte le immagini nella directory
    jpg_files = glob(f'{image_dir}/*.jpg')
    png_files = glob(f'{image_dir}/*.png')
    jpeg_files = glob(f'{image_dir}/*.jpeg')
    JPG_files = glob(f'{image_dir}/*.JPG')
    
    # Combina tutte le liste
    image_files = jpg_files + png_files + jpeg_files + JPG_files
    image_files = sorted(image_files)
    
    # Escludi il template se specificato
    if exclude_template:
        template_name = Path(exclude_template).name
        filtered_files = []
        for img_file in image_files:
            if Path(img_file).name != template_name:
                filtered_files.append(img_file)
        image_files = filtered_files
    
    # Seleziona casualmente n immagini
    random.seed(seed)
    num_to_select = min(n, len(image_files))
    selected = random.sample(image_files, num_to_select)
    
    print(f"Selezionate {len(selected)} immagini da {len(image_files)} disponibili (seed={seed})")
    
    return selected


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

    # Tengo per motivi didiattici
    target_edges = cv2.Canny(gray_blurred, 50, 150)
    
    # 3. Crea detector OpenCV Generalized Hough Ballard
    ght = cv2.createGeneralizedHoughBallard()
    
    # 4. Imposta il template
    ght.setTemplate(template_edges, template_center)
    
    # 5. Parametri del detector
    # Da ablation test
    # >> setVotesThreshold(20) + setMinDist(100) + setCannyLowThresh(10)
    ght.setMinDist(100)
    ght.setDp(2)
    ght.setCannyLowThresh(10)
    ght.setCannyHighThresh(30)
    ght.setLevels(360)
    ght.setVotesThreshold(20)
    
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
        'votes': votes,
        'detected': center is not None,
        'confidence': votes
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
            'n_inliers': 0,
            'detected': False,
            'confidence': 0
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
            'n_inliers': 0,
            'detected': False,
            'confidence': 0
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
            'n_inliers': 0,
            'detected': False,
            'confidence': n_matches
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
        'n_inliers': n_inliers,
        'detected': center is not None,
        'confidence': n_inliers
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
        'votes': votes,
        'detected': center is not None,
        'confidence': votes
    }


def build_r_table(edges, center, n_bins=36):
    """
    Costruisce una R-table semplificata dai bordi del template per GHT.
    
    Parametri:
        edges: bordi del template (immagine binaria)
        center: centro del template (x, y)
        n_bins: numero di bins per quantizzare gli angoli (default 36 = 10° per bin)
    
    Ritorna:
        dizionario: R-table con angle_bin -> lista di vettori R
    """
    # Calcola gradienti
    dy, dx = np.gradient(edges.astype(float))
    gradient_angle = np.arctan2(dy, dx)
    
    # Estrai punti di bordo e loro orientazioni
    edge_points = np.column_stack(np.where(edges > 0))
    r_table = {}
    
    for ep in edge_points:
        y = ep[0]
        x = ep[1]
        angle = gradient_angle[y, x]
        # Quantizza l'angolo
        angle_bin_float = angle / (2 * np.pi) * n_bins
        angle_bin = int(np.round(angle_bin_float) % n_bins)
        
        # Vettore R dal punto di bordo al centro
        r_vec = (center[0] - x, center[1] - y)
        
        if angle_bin not in r_table:
            r_table[angle_bin] = []
        r_table[angle_bin].append(r_vec)
    
    return r_table


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


def print_evaluation_report(eval_results, method_name):
    """
    Stampa un report essenziale della valutazione.
    
    Parametri:
        eval_results: risultato da evaluate_with_ground_truth()
        method_name: nome del metodo (es. 'GHT', 'SIFT', 'NCC')
    """
    # Stampa solo il nome del metodo in modo pulito
    pass


def compare_methods_with_ground_truth(image_paths, template_gray, ground_truth, 
                                      distance_threshold=100, standard_size=STANDARD_SIZE):
    """
    Confronta tutti e tre i metodi usando ground truth.
    
    Parametri:
        image_paths: lista di percorsi immagini da testare
        template_gray: immagine template in scala di grigi
        ground_truth: dict da load_ground_truth()
        distance_threshold: soglia distanza per TP (default 100px)
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
    print_evaluation_report(evaluation, 'GHT')
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
    print_evaluation_report(evaluation, 'SIFT')
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
    print_evaluation_report(evaluation, 'NCC')
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
    f1_scores = [comparison_results[m.lower()]['evaluation']['f1_score'] for m in methods]
    avg_times = [comparison_results[m.lower()]['avg_time'] * 1000 for m in methods]  # Convert to ms
    
    # Crea figura con 3 righe
    # Riga 1: Precision, Recall, F1
    # Riga 2: Tempo medio
    # Riga 3: Conteggi TP/FP/FN
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.4)
    
    # --- RIGA 1: Metriche (Precision, Recall, F1) ---
    ax_prec = fig.add_subplot(gs[0, 0])
    ax_rec = fig.add_subplot(gs[0, 1])
    ax_f1 = fig.add_subplot(gs[0, 2])
    
    # Precision
    ax_prec.bar(methods, precisions, color=colors, alpha=0.7, edgecolor='black')
    ax_prec.set_ylim(0, 1.05)
    ax_prec.set_title('Precision', fontsize=12, fontweight='bold')
    ax_prec.grid(axis='y', alpha=0.3)
    ax_prec.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    for i, p in enumerate(precisions):
        ax_prec.text(i, p + 0.02, f'{p:.1%}', ha='center', fontsize=10, fontweight='bold')
    
    # Recall
    ax_rec.bar(methods, recalls, color=colors, alpha=0.7, edgecolor='black')
    ax_rec.set_ylim(0, 1.05)
    ax_rec.set_title('Recall', fontsize=12, fontweight='bold')
    ax_rec.grid(axis='y', alpha=0.3)
    ax_rec.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    for i, r in enumerate(recalls):
        ax_rec.text(i, r + 0.02, f'{r:.1%}', ha='center', fontsize=10, fontweight='bold')
        
    # F1 Score
    ax_f1.bar(methods, f1_scores, color=colors, alpha=0.7, edgecolor='black')
    ax_f1.set_ylim(0, 1.05)
    ax_f1.set_title('F1 Score', fontsize=12, fontweight='bold')
    ax_f1.grid(axis='y', alpha=0.3)
    ax_f1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    for i, f in enumerate(f1_scores):
        ax_f1.text(i, f + 0.02, f'{f:.1%}', ha='center', fontsize=10, fontweight='bold')
    
    # --- RIGA 2: Tempo Medio Immagine ---
    ax_time = fig.add_subplot(gs[1, :])  # Occupa tutta la riga
    
    ax_time.bar(methods, avg_times, color=colors, alpha=0.7, edgecolor='black', width=0.4)
    ax_time.set_ylabel('Tempo (ms)', fontsize=11)
    max_time = max(avg_times) if avg_times else 1
    ax_time.set_ylim(0, max_time * 1.15)
    ax_time.set_title('Tempo Medio Elaborazione per Immagine', fontsize=12, fontweight='bold')
    ax_time.grid(axis='y', alpha=0.3)
    
    for i, t in enumerate(avg_times):
        ax_time.text(i, t + max_time * 0.02, f'{t:.1f} ms', ha='center', fontsize=10, fontweight='bold')
        
    # --- RIGA 3: Conteggi TP, FP, FN ---
    ax_counts_ght = fig.add_subplot(gs[2, 0])
    ax_counts_sift = fig.add_subplot(gs[2, 1])
    ax_counts_ncc = fig.add_subplot(gs[2, 2])
    
    axes_counts = [ax_counts_ght, ax_counts_sift, ax_counts_ncc]
    
    for i, method in enumerate(methods):
        ax = axes_counts[i]
        eval_data = comparison_results[method.lower()]['evaluation']
        
        # Matrice confusione
        tp = eval_data['true_positives']
        fp = eval_data['false_positives']
        fn = eval_data['false_negatives']
        
        categories = ['TP', 'FP', 'FN']
        values = [tp, fp, fn]
        bar_colors = ['green', 'red', 'orange']
        
        ax.bar(categories, values, color=bar_colors, alpha=0.7, edgecolor='black')
        ax.set_title(f'{method} - Conteggi', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Totale per scala y
        max_val = max(values) if values else 1
        ax.set_ylim(0, max([v for m in methods for v in [
            comparison_results[m.lower()]['evaluation']['true_positives'],
            comparison_results[m.lower()]['evaluation']['false_positives'],
            comparison_results[m.lower()]['evaluation']['false_negatives']
        ]]) * 1.15 + 1)
        
        for k, v in enumerate(values):
            ax.text(k, v + 0.2, str(v), ha='center', fontweight='bold')

    plt.suptitle('Benchmark Completo: GHT vs SIFT vs NCC', fontsize=16, y=0.95)
    plt.show()



# FUNZIONI PER IL NOTEBOOK ABLATION STUDY 
#
#

import pandas as pd
import json
import time
from itertools import product
from tqdm import tqdm

# Configurazioni globali per ablation study
DISTANCE_THRESHOLD = 100     # Soglia per matching con ground truth (pixel)
GAUSSIAN_KERNEL = (5, 5)    # Kernel per blur
GAUSSIAN_SIGMA = 1.4        # Sigma per blur
CANNY_HIGH_THRESH = 150     # Soglia alta Canny (fissa)
DP_VALUE = 2                # Parametro dp per GHT (fisso)
LEVELS_VALUE = 360          # Parametro levels per GHT (fisso)


def load_dataset(data_dir, template_path, gt_path):
    """
    Carica il dataset di immagini e ground truth.
    
    Args:
        data_dir: Path della directory dati
        template_path: Path del template
        gt_path: Path del file ground truth JSON
    
    Returns:
        tuple: (lista_immagini, dizionario_ground_truth, template_edges, template_center)
    """
    print("Caricamento dataset in corso...")
    
    # Carica ground truth
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Lista immagini disponibili
    image_files = list(data_dir.glob('*.jpg'))
    # Filtra solo quelle con ground truth
    valid_images = []
    for img_path in image_files:
        img_name = img_path.name
        if img_name in ground_truth:
            valid_images.append(img_path)
    
    print(f"Trovate {len(valid_images)} immagini valide con ground truth")
    
    # Carica e preprocessa template
    template_img = cv2.imread(str(template_path))
    if template_img is None:
        raise ValueError(f"Template non trovato: {template_path}")
    
    # Preprocessa template
    template_resized = cv2.resize(template_img, STANDARD_SIZE)
    template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
    template_blurred = cv2.GaussianBlur(template_gray, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    template_edges = cv2.Canny(template_blurred, 50, CANNY_HIGH_THRESH)
    
    # Calcola centro template
    edge_points = np.column_stack(np.where(template_edges > 0))
    if len(edge_points) > 0:
        center_y, center_x = np.mean(edge_points, axis=0).astype(int)
        template_center = (center_x, center_y)
    else:
        template_center = (STANDARD_SIZE[0]//2, STANDARD_SIZE[1]//2)
    
    print(f"Template processato. Centro: {template_center}")
    print(f"Pixel di bordo nel template: {np.count_nonzero(template_edges)}")
    
    return valid_images, ground_truth, template_edges, template_center


def run_ght_detection(image_path, template_edges, template_center, params):
    """
    Esegue detection GHT con parametri specificati.
    
    Args:
        image_path: percorso dell'immagine
        template_edges: bordi del template
        template_center: centro del template
        params: dizionario parametri {votes_threshold, min_dist, canny_low_thresh}
    
    Returns:
        dict: {'center': (x,y) or None, 'confidence': float, 'inference_time': float}
    """
    start_time = time.time()
    
    try:
        # Carica e preprocessa immagine
        img = cv2.imread(str(image_path))
        if img is None:
            return {'center': None, 'confidence': 0.0, 'inference_time': 0.0}
        
        img_resized = cv2.resize(img, STANDARD_SIZE)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
        edges = cv2.Canny(blurred, params['canny_low_thresh'], CANNY_HIGH_THRESH)
        
        # Crea detector GHT
        ght = cv2.createGeneralizedHoughBallard()
        
        # Imposta template
        ght.setTemplate(template_edges, template_center)
        
        # Imposta parametri
        ght.setMinDist(params['min_dist'])
        ght.setDp(DP_VALUE)
        ght.setCannyLowThresh(params['canny_low_thresh'])
        ght.setCannyHighThresh(CANNY_HIGH_THRESH)
        ght.setLevels(LEVELS_VALUE)
        ght.setVotesThreshold(params['votes_threshold'])
        
        # Rileva
        positions = ght.detect(edges)
        
        # Estrai risultato
        center = None
        confidence = 0.0
        
        if positions is not None and len(positions) > 0:
            if positions[0] is not None and len(positions[0]) > 0:
                pos = positions[0][0]
                pos_flat = np.array(pos).flatten()
                if len(pos_flat) >= 2:
                    center = (int(pos_flat[0]), int(pos_flat[1]))
                    confidence = float(len(positions[0]) * params['votes_threshold'])
        
        inference_time = time.time() - start_time
        
        return {
            'center': center,
            'confidence': confidence,
            'inference_time': inference_time
        }
        
    except Exception as e:
        print(f"Errore nell'elaborazione {image_path}: {e}")
        return {'center': None, 'confidence': 0.0, 'inference_time': time.time() - start_time}


def evaluate_detection(pred_center, gt_center, threshold=DISTANCE_THRESHOLD):
    """
    Valuta una detection confrontandola con il ground truth.
    
    Args:
        pred_center: centro predetto (x, y) o None
        gt_center: centro ground truth (x, y)
        threshold: soglia distanza per considerare match valido
    
    Returns:
        dict: {'is_tp': bool, 'distance': float}
    """
    if pred_center is None:
        # Nessuna detection -> False Negative
        return {'is_tp': False, 'distance': float('inf')}
    
    # Calcola distanza euclidea
    distance = np.sqrt((pred_center[0] - gt_center[0])**2 + 
                      (pred_center[1] - gt_center[1])**2)
    
    # Check se è True Positive
    is_tp = distance <= threshold
    
    return {'is_tp': is_tp, 'distance': distance}


def calculate_metrics(results_list):
    """
    Calcola metriche aggregate da lista di risultati.
    
    Args:
        results_list: lista di dict con 'is_tp', 'distance', 'inference_time'
    
    Returns:
        dict: metriche aggregate
    """
    if not results_list:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'mean_distance': float('inf'), 'std_distance': 0.0,
            'mean_inference_time': 0.0
        }
    
    # Conta TP, FP, FN
    tp = sum(1 for r in results_list if r['is_tp'])
    fp = sum(1 for r in results_list if not r['is_tp'] and r['distance'] != float('inf'))
    fn = sum(1 for r in results_list if r['distance'] == float('inf'))
    
    # Calcola metriche
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Distanze (solo per detection valide)
    valid_distances = [r['distance'] for r in results_list if r['distance'] != float('inf')]
    mean_distance = np.mean(valid_distances) if valid_distances else float('inf')
    std_distance = np.std(valid_distances) if len(valid_distances) > 1 else 0.0
    
    # Tempo inferenza
    inference_times = [r['inference_time'] for r in results_list]
    mean_inference_time = np.mean(inference_times)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'mean_inference_time': mean_inference_time,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def ablation_study(images, ground_truth, template_edges, template_center):
    """
    Esegue l'ablation study completo testando tutte le combinazioni di parametri.
    
    Args:
        images: lista percorsi immagini
        ground_truth: dizionario ground truth
        template_edges: bordi template
        template_center: centro template
    
    Returns:
        pd.DataFrame: risultati completi
    """
    # Definisci griglia parametri - SOLO un parametro variabile alla volta
    # Gli altri devono corrispondere ESATTAMENTE ai default usati in detect_starfish_ght:
    # min_dist=50, canny_low_thresh=50, votes_threshold=15
    
    base_params = {
        'votes_threshold': 15,
        'min_dist': 50,
        'canny_low_thresh': 50
    }
    
    param_grid = {
        'votes_threshold': [10, 15, 20, 25],
        'min_dist': [30, 50, 70],
        'canny_low_thresh': [30, 50, 70]
    }
    
    # Genera configurazioni variando un parametro alla volta
    param_combinations = []
    
    # 1. Varia solo votes_threshold
    for v in param_grid['votes_threshold']:
        params = base_params.copy()
        params['votes_threshold'] = v
        param_combinations.append((params['votes_threshold'], params['min_dist'], params['canny_low_thresh']))
        
    # 2. Varia solo min_dist (escluso default già aggiunto)
    for v in param_grid['min_dist']:
        if v == 50: continue
        params = base_params.copy()
        params['min_dist'] = v
        param_combinations.append((params['votes_threshold'], params['min_dist'], params['canny_low_thresh']))
        
    # 3. Varia solo canny_low_thresh (escluso default già aggiunto)
    for v in param_grid['canny_low_thresh']:
        if v == 50: continue
        params = base_params.copy()
        params['canny_low_thresh'] = v
        param_combinations.append((params['votes_threshold'], params['min_dist'], params['canny_low_thresh']))
    
    # Rimuovi duplicati se ce ne sono
    param_combinations = sorted(list(set(param_combinations)))
    
    print(f"Testando {len(param_combinations)} configurazioni su {len(images)} immagini")
    print(f"Test totali: {len(param_combinations) * len(images)}")
    
    results = []
    
    # Progress bar per configurazioni
    for votes_th, min_d, canny_low in tqdm(param_combinations, desc="Configurazioni"):
        params = {
            'votes_threshold': votes_th,
            'min_dist': min_d,
            'canny_low_thresh': canny_low
        }
        
        # Risultati per questa configurazione
        config_results = []
        
        # Testa su tutte le immagini
        for img_path in images:
            img_name = img_path.name
            
            # Ottieni ground truth
            gt_data = ground_truth[img_name]
            gt_center = tuple(gt_data['center'])  # center è un array [x, y]
            
            # Esegui detection
            detection_result = run_ght_detection(img_path, template_edges, template_center, params)
            
            # Valuta risultato
            evaluation = evaluate_detection(detection_result['center'], gt_center)
            
            # Combina risultati
            combined_result = {
                'is_tp': evaluation['is_tp'],
                'distance': evaluation['distance'],
                'inference_time': detection_result['inference_time']
            }
            config_results.append(combined_result)
        
        # Calcola metriche per questa configurazione
        metrics = calculate_metrics(config_results)
        
        # Salva risultato
        result_row = {
            'votes_threshold': votes_th,
            'min_dist': min_d,
            'canny_low_thresh': canny_low,
            **metrics
        }
        results.append(result_row)
        
        # Log progresso
        if len(results) % 10 == 0:
            print(f"Completate {len(results)}/{len(param_combinations)} configurazioni")
    
    # Converti in DataFrame
    df_results = pd.DataFrame(results)
    
    # Ordina per F1-score decrescente
    df_results = df_results.sort_values('f1_score', ascending=False).reset_index(drop=True)
    
    print("\nStudio di ablazione completato!")
    print(f"Configurazioni testate: {len(df_results)}")
    
    return df_results
