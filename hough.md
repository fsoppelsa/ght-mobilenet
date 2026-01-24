# Trasformata di Hough: Dalla Teoria alla Pratica

## Indice
1. [Il Problema](#il-problema)
2. [Come la Trasformata di Hough Affronta il Problema](#come-la-trasformata-di-hough-affronta-il-problema)
3. [Storia e Motivazione](#storia-e-motivazione)
4. [Trasformata di Hough per Linee](#trasformata-di-hough-per-linee)
5. [Trasformata di Hough per Cerchi](#trasformata-di-hough-per-cerchi)
6. [Trasformata di Hough Generalizzata (GHT)](#trasformata-di-hough-generalizzata-ght)
7. [Implementazione e Ottimizzazioni](#implementazione-e-ottimizzazioni)
8. [Librerie e Implementazioni SOTA](#librerie-e-implementazioni-sota)
9. [Metodi Alternativi e Correlati](#metodi-alternativi-e-correlati)

---

## Il Problema

### Rilevamento di Forme in Immagini

Il rilevamento di forme geometriche (linee, cerchi, ellissi, forme arbitrarie) in immagini digitali è una delle sfide fondamentali della computer vision.

### Difficoltà Principali

| Problema | Descrizione | Conseguenza |
|----------|-------------|-------------|
| **Rumore** | Pixel spuri nei bordi | Falsi positivi |
| **Occlusioni** | Forme parzialmente visibili | Forme incomplete |
| **Discontinuità** | Bordi frammentati | Difficoltà nel collegare segmenti |
| **Prospettiva** | Distorsione geometrica | Forme deformate |
| **Illuminazione** | Variazioni di luminosità | Bordi mancanti o spuri |

### Esempio: Linee in Immagini

**Approccio naive - Fitting diretto:**
```
Per ogni coppia di punti di bordo (p1, p2):
    Calcola la linea passante per p1 e p2
    Verifica quanti altri punti giacciono sulla linea
```

**Complessità:** $O(n^2)$ per $n$ punti, ingestibile per immagini reali con migliaia di edge points.

### Punti di Fuga e Prospettiva

Nella visione prospettica, linee parallele nel mondo 3D convergono in **punti di fuga** nell'immagine 2D:

```
Mondo 3D (linee parallele)          Immagine 2D (convergenti)
    |     |     |                         \   |   /
    |     |     |            →             \  |  /
    |     |     |                           \ | /
    |     |     |                            \|/
                                              VP (vanishing point)
```

**Applicazioni:**
- Guida autonoma (rilevamento corsie)
- Architettura (analisi prospettica)
- Robotica (navigazione)
- Realtà aumentata (calibrazione camera)

---

## Come la Trasformata di Hough Affronta il Problema

### Idea Fondamentale: Dualità Punto-Curva

La trasformata di Hough sfrutta una **dualità geometrica** tra lo spazio immagine e lo spazio dei parametri.

**Principio chiave:**
> Un punto nello spazio immagine corrisponde a una curva nello spazio dei parametri.
> Una forma nello spazio immagine corrisponde a un punto nello spazio dei parametri.

### Meccanismo di Voto (Voting)

```
Spazio Immagine                    Spazio Parametri (Accumulatore)
      ×  (punto p1)      →         curva di tutti i parametri
                                   che passano per p1
      
      × × × (punti su linea)  →    intersezione delle curve
                                   = picco nell'accumulatore
```

**Algoritmo generale:**

1. **Discretizzazione**: Crea una griglia (accumulatore) nello spazio dei parametri
2. **Voting**: Per ogni punto di bordo, incrementa le celle corrispondenti ai parametri compatibili
3. **Peak Detection**: Trova i massimi locali nell'accumulatore
4. **Estrazione**: Converti i picchi in parametri della forma

### Vantaggi del Metodo

| Vantaggio | Spiegazione |
|-----------|-------------|
| **Robusto al rumore** | I voti spuri si distribuiscono, i voti reali si accumulano |
| **Gestisce occlusioni** | Forme parziali producono comunque picchi |
| **Parallelizzabile** | Ogni punto vota indipendentemente |
| **Globale** | Non richiede inizializzazione locale |

---

## Storia e Motivazione

### Timeline

| Anno | Evento | Autore |
|------|--------|--------|
| 1962 | Brevetto originale per rilevamento linee | Paul Hough (fisico delle particelle) |
| 1972 | Formulazione $(\rho, \theta)$ per linee | Duda & Hart |
| 1981 | Generalizzazione a forme arbitrarie (GHT) | Dana H. Ballard |
| 1986 | Ottimizzazioni per cerchi | Atherton & Kerbyson |
| 1999 | Probabilistic Hough Transform | Kiryati et al. |
| 2000s | Accelerazione GPU | Vari ricercatori |
| 2010s | Deep Hough Transform | Integrazione con CNN |

### Motivazione Originale

Paul Hough era un fisico che lavorava al **CERN** sulla rilevazione di particelle subatomiche nelle camere a bolle. Il problema era identificare le **traiettorie rettilinee** delle particelle tra migliaia di punti rumorosi.

**Contesto fisico:**
- Camere a bolle producono tracce di particelle
- Ogni traccia è approssimativamente una linea retta
- Milioni di punti da analizzare
- Necessità di metodi automatici

---

## Trasformata di Hough per Linee

### Rappresentazione Parametrica

#### Forma Slope-Intercept (problematica)

Una linea può essere rappresentata come:
$$y = mx + c$$

**Problema:** Linee verticali hanno $m = \infty$

#### Forma Normale (Duda-Hart, 1972)

Rappresentazione **polare** della linea:
$$\rho = x \cos\theta + y \sin\theta$$

Dove:
- $\rho$ = distanza dall'origine alla linea (può essere negativo)
- $\theta$ = angolo della normale alla linea rispetto all'asse x

```
                    linea
                      /
                     /
           ρ        /
    O ───────────►/   ← punto più vicino all'origine
                 /
                / θ
               /_______ asse x
```

**Vantaggi:**
- $\theta \in [0, \pi)$ (o $[0, 2\pi)$)
- $\rho \in [-D, D]$ dove $D$ è la diagonale dell'immagine
- Nessuna singolarità

### Algoritmo Standard

```python
def hough_lines(edge_image, rho_res=1, theta_res=np.pi/180):
    """
    Trasformata di Hough per linee
    
    Args:
        edge_image: immagine binaria dei bordi
        rho_res: risoluzione di rho (pixel)
        theta_res: risoluzione di theta (radianti)
    
    Returns:
        accumulator: matrice dei voti
        rhos: valori di rho
        thetas: valori di theta
    """
    h, w = edge_image.shape
    
    # Diagonale dell'immagine
    diag = int(np.sqrt(h**2 + w**2))
    
    # Discretizzazione dello spazio dei parametri
    thetas = np.arange(0, np.pi, theta_res)
    rhos = np.arange(-diag, diag, rho_res)
    
    # Accumulatore
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    
    # Precalcola sin e cos
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    
    # Trova i punti di bordo
    edge_points = np.argwhere(edge_image > 0)
    
    # Voting
    for y, x in edge_points:
        for t_idx, (cos_t, sin_t) in enumerate(zip(cos_thetas, sin_thetas)):
            rho = x * cos_t + y * sin_t
            r_idx = int((rho + diag) / rho_res)
            accumulator[r_idx, t_idx] += 1
    
    return accumulator, rhos, thetas
```

### Complessità

| Operazione | Complessità |
|------------|-------------|
| Spazio accumulatore | $O(n_\rho \times n_\theta)$ |
| Voting | $O(n_{edge} \times n_\theta)$ |
| Peak detection | $O(n_\rho \times n_\theta)$ |

Per un'immagine 1000×1000 con $n_\theta = 180$:
- $n_\rho \approx 2 \times 1414 = 2828$
- Accumulatore: ~500K celle
- Se 10K edge points: 1.8M operazioni di voto

### Dualità Geometrica Visualizzata

**Un punto $(x_0, y_0)$ genera una sinusoide nello spazio $(\theta, \rho)$:**

$$\rho = x_0 \cos\theta + y_0 \sin\theta$$

```
Spazio Immagine              Spazio Hough (θ, ρ)
                                    ρ
    × P1                           ╱╲
                                  ╱  ╲   ← sinusoide per P1
    × P2         →               ╱    ╲
                                ╱      ╲
    × P3 (sulla stessa linea)   ──────────── θ
                                   ↑
                              intersezione = linea rilevata
```

---

## Trasformata di Hough per Cerchi

### Rappresentazione Parametrica

Un cerchio è definito da:
$$(x - a)^2 + (y - b)^2 = r^2$$

Parametri: $(a, b, r)$ - centro e raggio

### Spazio 3D dei Parametri

**Problema:** L'accumulatore è tridimensionale!

Per un'immagine 500×500 con raggi $r \in [10, 100]$:
- Accumulatore: $500 \times 500 \times 90 = 22.5M$ celle

### Ottimizzazione con Gradiente

**Idea:** Usare la direzione del gradiente per ridurre la dimensionalità.

Il gradiente in un punto di bordo **punta verso il centro** del cerchio:
$$\vec{g}(x, y) = \nabla I(x, y)$$

**Algoritmo ottimizzato:**

```python
def hough_circles_gradient(edge_image, gradient_dir, r_min, r_max):
    """
    Hough per cerchi usando direzione del gradiente
    """
    h, w = edge_image.shape
    
    # Per ogni raggio candidato
    for r in range(r_min, r_max + 1):
        accumulator = np.zeros((h, w), dtype=np.int32)
        
        for y, x in np.argwhere(edge_image > 0):
            # Direzione del gradiente
            theta = gradient_dir[y, x]
            
            # Vota solo lungo la direzione del gradiente
            # Centro candidato in direzione del gradiente
            a1 = int(x - r * np.cos(theta))
            b1 = int(y - r * np.sin(theta))
            
            # Centro candidato in direzione opposta
            a2 = int(x + r * np.cos(theta))
            b2 = int(y + r * np.sin(theta))
            
            if 0 <= a1 < w and 0 <= b1 < h:
                accumulator[b1, a1] += 1
            if 0 <= a2 < w and 0 <= b2 < h:
                accumulator[b2, a2] += 1
        
        # Trova picchi per questo raggio
        detect_peaks(accumulator, r)
```

**Riduzione:** Da $O(n \times n_\theta)$ a $O(n \times 2)$ per il voting!

---

## Trasformata di Hough Generalizzata (GHT)

### Paper Originale (Ballard, 1981)
*"Generalizing the Hough Transform to Detect Arbitrary Shapes"*

### Idea Chiave: R-Table

La GHT può rilevare **qualsiasi forma** usando una tabella di riferimento (R-Table) costruita da un template.

**Concetto:**
> Memorizza, per ogni orientazione del gradiente sul bordo del template, 
> il vettore che punta dal punto di bordo a un punto di riferimento (es. centroide).

### Costruzione della R-Table

```
Template con punto di riferimento (xc, yc)
          
          ★ ← punto di riferimento (centroide)
         /|\
        / | \
       /  |  \
      ×   |   ×  ← punti di bordo
       \  |  /
        \ | /
         \|/
          
Per ogni punto di bordo (x, y):
  1. Calcola orientazione gradiente φ
  2. Calcola vettore r = (xc - x, yc - y)
  3. Memorizza r nella R-table[φ]
```

**Struttura R-Table:**

| Angolo φ (quantizzato) | Vettori r |
|------------------------|-----------|
| 0° | [(3, 5), (2, 4), ...] |
| 10° | [(4, -2), ...] |
| 20° | [(-1, 6), (0, 7), ...] |
| ... | ... |
| 350° | [(5, 1), ...] |

### Algoritmo di Detection

```python
class GeneralizedHoughTransform:
    def __init__(self, template_edges, reference_point):
        """
        Costruisce la R-Table dal template
        """
        self.r_table = {}
        
        # Calcola gradiente del template
        dy, dx = np.gradient(template_edges.astype(float))
        gradient_angle = np.arctan2(dy, dx)
        
        # Per ogni punto di bordo
        for y, x in np.argwhere(template_edges > 0):
            phi = gradient_angle[y, x]
            # Quantizza l'angolo (es. 36 bins = 10° ciascuno)
            phi_idx = int(np.round(phi / (2 * np.pi) * 36) % 36)
            
            # Vettore al punto di riferimento
            r_vec = (reference_point[0] - x, reference_point[1] - y)
            
            if phi_idx not in self.r_table:
                self.r_table[phi_idx] = []
            self.r_table[phi_idx].append(r_vec)
    
    def detect(self, target_edges):
        """
        Rileva la forma nell'immagine target
        """
        h, w = target_edges.shape
        accumulator = np.zeros((h, w), dtype=np.int32)
        
        # Calcola gradiente del target
        dy, dx = np.gradient(target_edges.astype(float))
        gradient_angle = np.arctan2(dy, dx)
        
        # Voting
        for y, x in np.argwhere(target_edges > 0):
            phi = gradient_angle[y, x]
            phi_idx = int(np.round(phi / (2 * np.pi) * 36) % 36)
            
            # Recupera vettori dalla R-table
            if phi_idx in self.r_table:
                for r_vec in self.r_table[phi_idx]:
                    # Calcola posizione candidata del centro
                    xc = x + r_vec[0]
                    yc = y + r_vec[1]
                    
                    if 0 <= xc < w and 0 <= yc < h:
                        accumulator[int(yc), int(xc)] += 1
        
        return accumulator
```

### Gestione di Scala e Rotazione

Per gestire variazioni di **scala** e **rotazione**, l'accumulatore diventa 4D:

$$A(x_c, y_c, s, \theta)$$

Dove:
- $(x_c, y_c)$ = posizione del centro
- $s$ = fattore di scala
- $\theta$ = angolo di rotazione

**Durante il voting:**
```python
for s in scale_range:
    for theta in rotation_range:
        # Ruota e scala il vettore r
        r_transformed = rotate_scale(r_vec, theta, s)
        xc = x + r_transformed[0]
        yc = y + r_transformed[1]
        accumulator[yc, xc, s_idx, theta_idx] += 1
```

**Complessità:** $O(n_{edge} \times n_\phi \times n_s \times n_\theta)$

### Esempio: Rilevamento Stelle Marine

```
Template (stella marina)              Target Image
                                      
      ★                               ╔═══════════════╗
     /|\                              ║     ★         ║
    / | \                             ║    /|\        ║
   ×  |  ×           →                ║   / | \   ?   ║
    \ | /                             ║  ×  |  ×      ║
     \|/                              ║   \ | /       ║
                                      ║    \|/        ║
                                      ╚═══════════════╝
                                      
R-Table costruita    →    Voting    →    Picco nell'accumulatore
dal template                             = posizione della stella
```

---

## Implementazione e Ottimizzazioni

### 1. Probabilistic Hough Transform (PHT)

**Problema:** La Hough standard è computazionalmente costosa.

**Soluzione:** Campiona casualmente un sottoinsieme di punti di bordo.

```python
def probabilistic_hough_lines(edges, threshold, min_line_length, max_line_gap):
    """
    Progressive Probabilistic Hough Transform (PPHT)
    """
    # 1. Estrai tutti i punti di bordo
    edge_points = list(np.argwhere(edges > 0))
    random.shuffle(edge_points)
    
    lines = []
    
    while edge_points:
        # 2. Seleziona un punto casuale
        y, x = edge_points.pop()
        
        # 3. Vota solo per questo punto
        # 4. Trova il bin con più voti
        # 5. Cerca altri punti sulla stessa linea
        # 6. Se abbastanza punti, aggiungi la linea
        # 7. Rimuovi i punti usati
        
    return lines
```

**Vantaggi:**
- 10-100x più veloce
- Restituisce segmenti invece di linee infinite
- Implementato in `cv2.HoughLinesP()`

### 2. Hierarchical Hough Transform

**Idea:** Usa una piramide di risoluzioni dell'accumulatore.

```
Risoluzione Bassa        Media            Alta
┌───────────┐       ┌───────────┐    ┌───────────┐
│  █        │   →   │  █ █      │ →  │ █ █ █     │
│           │       │  █        │    │ █ █       │
│           │       │           │    │ █         │
└───────────┘       └───────────┘    └───────────┘
Trova regioni       Raffina          Localizzazione
candidate           la ricerca       precisa
```

### 3. Kernel-Based Hough Transform (KHT)

**Innovazione:** Usa cluster di punti invece di punti singoli.

1. Raggruppa punti di bordo vicini in segmenti
2. Ogni segmento vota con un kernel gaussiano
3. Riduce il rumore nell'accumulatore

### 4. Accelerazione GPU (CUDA)

```python
# Pseudo-codice CUDA per Hough parallelo
@cuda.jit
def hough_kernel(edges, accumulator, cos_table, sin_table):
    # Ogni thread processa un punto di bordo
    idx = cuda.grid(1)
    
    if idx < len(edge_points):
        x, y = edge_points[idx]
        
        for t in range(n_theta):
            rho = x * cos_table[t] + y * sin_table[t]
            r_idx = int((rho + diag) / rho_res)
            
            # Atomic add per evitare race conditions
            cuda.atomic.add(accumulator, (r_idx, t), 1)
```

**Speedup:** 50-200x rispetto a CPU

### 5. Deep Hough Transform

Integrazione con reti neurali per:
- Apprendere features migliori per il voting
- Predire direttamente i parametri delle forme
- Gestire forme deformabili

```
Image → CNN Feature Extractor → Hough Voting Layer → Output
                 ↓
         Features apprese         Voting differenziabile
         (invece di edge pixels)  (backpropagation possibile)
```

---

## Librerie e Implementazioni SOTA

### 1. OpenCV

```python
import cv2
import numpy as np

# Carica immagine
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 50, 150)

# ===== LINEE =====

# Standard Hough Transform
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

# Probabilistic Hough Transform (più veloce, restituisce segmenti)
lines_p = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                          threshold=50, minLineLength=50, maxLineGap=10)

# ===== CERCHI =====

# Hough Circle Transform (metodo del gradiente)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, 
                           minDist=50, param1=100, param2=30,
                           minRadius=10, maxRadius=100)

# HOUGH_GRADIENT_ALT (più preciso, OpenCV 4.x)
circles_alt = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, dp=1.5,
                               minDist=50, param1=300, param2=0.9,
                               minRadius=10, maxRadius=100)

# ===== GENERALIZED HOUGH =====

# Ballard method (traslazione)
ght_ballard = cv2.createGeneralizedHoughBallard()
ght_ballard.setTemplate(template_edges)
ght_ballard.setMinDist(50)
ght_ballard.setLevels(360)
ght_ballard.setVotesThreshold(100)
positions = ght_ballard.detect(target_edges)

# Guil method (traslazione + scala + rotazione)
ght_guil = cv2.createGeneralizedHoughGuil()
ght_guil.setTemplate(template_edges)
ght_guil.setMinScale(0.5)
ght_guil.setMaxScale(2.0)
ght_guil.setScaleStep(0.1)
ght_guil.setMinAngle(-180)
ght_guil.setMaxAngle(180)
ght_guil.setAngleStep(1)
positions = ght_guil.detect(target_edges)
```

### 2. scikit-image

```python
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

# Edge detection
edges = canny(image, sigma=2)

# ===== LINEE =====

# Hough Transform
h, theta, d = hough_line(edges)

# Peak detection
accum, angles, dists = hough_line_peaks(h, theta, d, num_peaks=10)

# Probabilistic Hough
from skimage.transform import probabilistic_hough_line
lines = probabilistic_hough_line(edges, threshold=10, 
                                  line_length=50, line_gap=3)

# ===== CERCHI =====

# Range di raggi da testare
hough_radii = np.arange(20, 100, 2)

# Hough Transform per cerchi
hough_res = hough_circle(edges, hough_radii)

# Peak detection
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, 
                                            total_num_peaks=10)
```

### 3. MATLAB Image Processing Toolbox

```matlab
% Carica immagine e rileva bordi
I = imread('image.jpg');
BW = edge(rgb2gray(I), 'canny');

% ===== LINEE =====

% Standard Hough
[H, theta, rho] = hough(BW);

% Peak detection
P = houghpeaks(H, 10, 'threshold', ceil(0.3*max(H(:))));

% Estrai segmenti di linea
lines = houghlines(BW, theta, rho, P, 'FillGap', 5, 'MinLength', 50);

% ===== CERCHI =====

[centers, radii, metric] = imfindcircles(I, [20 100], ...
    'ObjectPolarity', 'bright', 'Sensitivity', 0.9);
```

### 4. Implementazioni Deep Learning

#### Deep Hough Transform (DHT)

```python
# Repository: https://github.com/Hanqer/deep-hough-transform
import torch
from dht import DeepHoughTransform

model = DeepHoughTransform(backbone='resnet50')
model.load_state_dict(torch.load('dht_pretrained.pth'))

# Inference
with torch.no_grad():
    lines = model(image_tensor)
```

#### LETR (Line Segment Detection with Transformers)

```python
# Repository: https://github.com/mlpc-ucsd/LETR
from models import build_model

model = build_model(args)
outputs = model(image_tensor)
# outputs contiene linee predette con attention
```

### 5. Implementazione GPU Custom (CuPy/Numba)

```python
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates

def hough_lines_gpu(edges_gpu, n_theta=180):
    """
    Hough Transform accelerata su GPU con CuPy
    """
    h, w = edges_gpu.shape
    diag = int(cp.sqrt(h**2 + w**2))
    
    # Crea accumulatore su GPU
    accumulator = cp.zeros((2*diag, n_theta), dtype=cp.int32)
    
    # Tabelle trigonometriche
    thetas = cp.linspace(0, cp.pi, n_theta, endpoint=False)
    cos_t = cp.cos(thetas)
    sin_t = cp.sin(thetas)
    
    # Trova punti di bordo
    ys, xs = cp.where(edges_gpu > 0)
    
    # Voting vettorizzato
    for t_idx in range(n_theta):
        rhos = (xs * cos_t[t_idx] + ys * sin_t[t_idx]).astype(cp.int32)
        rhos += diag
        
        # Accumula voti
        cp.add.at(accumulator[:, t_idx], rhos, 1)
    
    return accumulator
```

---

## Metodi Alternativi e Correlati

### 1. RANSAC (Random Sample Consensus)

**Principio:** Campiona casualmente punti minimi per stimare il modello, poi valuta il consenso.

```python
def ransac_line(points, n_iterations=1000, threshold=5.0):
    """
    RANSAC per fitting di linee
    """
    best_line = None
    best_inliers = 0
    
    for _ in range(n_iterations):
        # 1. Campiona 2 punti casuali
        sample = random.sample(points, 2)
        
        # 2. Calcola la linea passante per i 2 punti
        line = fit_line(sample)
        
        # 3. Conta gli inliers
        inliers = count_points_near_line(points, line, threshold)
        
        # 4. Aggiorna il miglior modello
        if inliers > best_inliers:
            best_inliers = inliers
            best_line = line
    
    return best_line
```

**Confronto Hough vs RANSAC:**

| Aspetto | Hough | RANSAC |
|---------|-------|--------|
| Tipo | Deterministico | Probabilistico |
| Output | Tutti i modelli sopra soglia | Miglior modello singolo |
| Complessità | $O(n \times n_{params})$ | $O(k \times n)$ |
| Forme multiple | ✓ Naturale | ✗ Richiede iterazioni |
| Robusto outliers | ✓ | ✓✓ |

### 2. Least Squares Fitting

Per forme con molti inliers, il fitting ai minimi quadrati è ottimale:

$$\min_\theta \sum_i d(p_i, f(\theta))^2$$

**Problema:** Sensibile agli outliers.

### 3. M-Estimators

Robusti agli outliers usando funzioni di costo diverse:

$$\min_\theta \sum_i \rho(d(p_i, f(\theta)))$$

Dove $\rho$ è una funzione robusta (Huber, Tukey, ecc.)

### 4. LSD (Line Segment Detector)

**Approccio:** Raggruppa pixel con gradienti allineati.

```python
import cv2

# LSD è integrato in OpenCV
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
lines, width, prec, nfa = lsd.detect(gray_image)

# Visualizzazione
drawn_img = lsd.drawSegments(image.copy(), lines)
```

**Vantaggi rispetto a Hough:**
- Non richiede soglie
- Più veloce
- Fornisce larghezza e precisione del segmento

### 5. EDLines (Edge Drawing Lines)

Algoritmo ultra-veloce per segmenti:

```python
# Repository: https://github.com/CihanToworke/ED_Lib
import pyED

ed = pyED.EdgeDrawing()
ed.params.MinLineLength = 10
ed.params.MinPathLength = 10

segments = ed.detectLines(gray_image)
```

### 6. Confronto Complessivo Metodi per Linee

| Metodo | Velocità | Accuratezza | Robustezza | Parametri |
|--------|----------|-------------|------------|-----------|
| Hough Standard | ★★☆☆ | ★★★★ | ★★★★ | 3 |
| Hough Probabilistico | ★★★☆ | ★★★☆ | ★★★☆ | 5 |
| RANSAC | ★★★☆ | ★★★☆ | ★★★★★ | 3 |
| LSD | ★★★★★ | ★★★★ | ★★★☆ | 1 |
| EDLines | ★★★★★ | ★★★☆ | ★★★☆ | 2 |
| Deep Hough | ★★☆☆ | ★★★★★ | ★★★★ | Learned |

### 7. Applicazioni Moderne

| Applicazione | Metodo Preferito | Motivazione |
|--------------|------------------|-------------|
| Lane detection (auto) | Hough + CNN | Robusto, real-time |
| Document scanning | LSD + RANSAC | Veloce, preciso |
| Pupil detection | Hough Circles | Forma nota |
| Object detection | GHT + Deep Learning | Forme arbitrarie |
| Industrial inspection | Hough standard | Deterministico |

---

## Conclusioni

La trasformata di Hough rimane uno degli algoritmi fondamentali della computer vision dopo oltre 60 anni dalla sua invenzione:

1. **Hough per Linee**: Base per lane detection, document analysis
2. **Hough per Cerchi**: Rilevamento occhi, monete, oggetti circolari
3. **GHT**: Template matching robusto per forme arbitrarie
4. **Varianti moderne**: Integrazione con deep learning per accuratezza superiore

**Quando usare Hough:**
- Forme parametriche note (linee, cerchi, ellissi)
- Robusto a occlusioni e rumore
- Necessità di trovare tutte le istanze

**Quando NON usare Hough:**
- Forme deformabili
- Real-time con risorse limitate (preferire LSD/EDLines)
- Singola istanza di forma (preferire RANSAC)

---

## Riferimenti

1. Hough, P.V.C. "Method and means for recognizing complex patterns." U.S. Patent 3,069,654 (1962)
2. Duda, R.O. & Hart, P.E. "Use of the Hough transformation to detect lines and curves in pictures." Communications of the ACM (1972)
3. Ballard, D.H. "Generalizing the Hough Transform to Detect Arbitrary Shapes." Pattern Recognition (1981)
4. Kiryati, N. et al. "A probabilistic Hough transform." Pattern Recognition (1991)
5. Matas, J. et al. "Robust Detection of Lines Using the Progressive Probabilistic Hough Transform." CVIU (2000)
6. Xu, Y. et al. "Deep Hough Transform for Semantic Line Detection." ECCV (2020)
7. von Gioi, R.G. et al. "LSD: A Fast Line Segment Detector with a False Detection Control." TPAMI (2010)
