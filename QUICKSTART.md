# Magic Mirror Face Filter - Rýchla Príručka

## 🚀 Rýchly Štart

```bash
# Na Raspberry Pi ARM64 (Debian 13+)
sudo apt update
sudo apt install -y python3-opencv python3-numpy
pip3 install --user -r requirements.txt

# Spustenie
python3 main.py --windowed
```

## 🎮 Ovládanie

| Klávesa | Akcia |
|---------|-------|
| **SPACE** | Ďalší filter |
| **B** | Predchádzajúci filter |
| **Q** / **ESC** | Ukončiť |

## 📁 Štruktúra Projektu

```
Navesnak/
├── main.py           # Hlavná aplikácia
├── face_detector.py  # OpenCV DNN detekcia tváre
├── filters.py        # Konfigurácia a overlay filtrov
├── requirements.txt  # Python závislosti
├── models/           # DNN modely (stiahnu sa automaticky)
└── assets/filters/   # PNG obrázky filtrov
```

## 🔧 Ako to Funguje

### 1. Inicializácia
- `main.py` → `MagicMirrorApp.__init__()` inicializuje kameru a face detector

### 2. Detekcia Tváre
- `face_detector.py` → OpenCV DNN SSD model detekuje tvár
- `FacemarkLBF` extrahuje 68 landmarkov (oči, nos, čelo, ústa)

### 3. Aplikácia Filtrov
- `filters.py` → `FilterManager` načíta PNG obrázky
- Automaticky odstráni biele pozadie
- `overlay_filter()` aplikuje filter na anchor point (nos, oči, čelo)

### 4. Hlavná Slučka
```
while True:
    1. Načítať frame z kamery
    2. Zrkadlovo prevrátiť (cv2.flip)
    3. Detekovať tvár a landmarky
    4. Aplikovať aktuálny filter
    5. Zobraziť frame
    6. Spracovať klávesnicu (SPACE/B/Q)
```

## 📦 Dostupné Filtre

| Filter | Anchor Point | Popis |
|--------|-------------|-------|
| Mustache | `nose` | Fúzy pod nosom |
| Glasses | `eyes_center` | Okuliare na očiach |
| Clown Nose | `nose` | Červený klaunský nos |
| Unicorn Horn | `forehead` | Rozprávkový roh na čele |

## 🛠 Pridanie Nového Filtra

1. Pridaj PNG obrázok do `assets/filters/`
2. Uprav `filters.py`, pridaj do `filter_defs`:
```python
{
    "name": "Moj Filter",
    "image": "moj_filter.png",
    "anchor": "forehead",  # nose, eyes_center, forehead, mouth
    "scale_factor": 1.0,   # veľkosť relatívna k šírke tváre
    "offset_y": -50,       # posun (záporné = hore)
}
```

## 🐛 Problémy

**Kamera nefunguje:**
```bash
ls /dev/video*
python3 main.py --camera-index 1
```

**Modely sa nestiahli:**
```bash
mkdir -p models && cd models
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
wget https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
```
