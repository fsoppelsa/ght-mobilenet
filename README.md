# Computer Vision: Metodi Classici e Deep Learning

Questo repository contiene un'analisi comparativa di metodi di computer vision classici e deep sviluppata nell'ambito del corso di Visione Artificiale presso l'Università degli Studi di Palermo (A.A. 2025/2026).

## Obiettivo del Progetto

Il progetto si propone di confrontare quantitativamente l'efficacia di approcci tradizionali e moderni di computer vision attraverso implementazioni pratiche e benchmark. 

Vengono analizzati tre scenari principali: il rilevamento di oggetti mediante metodi classici (Generalized Hough Transform, SIFT con RANSAC, Template Matching), la classificazione di immagini utilizzando architetture CNN efficienti (MobileNet v1-v4, SqueezeNet, GhostNet), e l'object detection in tempo reale su video. 

'obiettivo è comprendere i trade-off tra accuratezza, velocità di elaborazione e complessità computazionale per applicazioni reali, incluso il deployment su dispositivi embedded e mobile.

## Struttura

I file principali includono: `hough.md` e `hough.ipynb` (teoria e implementazione della Trasformata di Hough), `mobilenet.md` e `mobilenet.ipynb` (analisi delle architetture MobileNet per classificazione), `video_detection.ipynb` (applicazione di object detection su video).

## Risultati Principali

Per i **metodi classici**, la Generalized Hough Transform ha ottenuto la migliore recall (87%) nel rilevamento di stelle marine, Template Matching (NCC) si è dimostrato il più veloce (95ms per immagine), mentre SIFT ha fornito la localizzazione più precisa grazie alla verifica geometrica.

Nel **benchmark deep learning**, MobileNetV4 ha raggiunto l'accuratezza top-1 più elevata (74.6% su ImageNet), MobileNetV3-Small ha offerto il miglior compromesso efficienza/prestazioni (67.4% con solo 2.5M parametri), e MobileNetV1 si è confermato la scelta ottimale per risorse estremamente limitate.

Per l'**object detection su video**, il sistema ha raggiunto circa 6.5 FPS su GPU NVIDIA con rilevamento accurato di oggetti multipli su dataset COCO, dimostrando capacità di elaborazione real-time con architetture leggere ottimizzate per deployment mobile.