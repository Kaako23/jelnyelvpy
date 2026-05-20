# MediaPipe → modell bemenet (jellemzők)

Ez a fájl rögzíti, **mit** veszünk ki a MediaPipe Tasks API-ból, **hogyan** pakoljuk vektorba, és **milyen skálákkal** megy a tanítás/felismerés felé. Hibakeresésnél (pl. egykezes jel, rossz osztály) ide érdemes visszanézni.

## Használt MediaPipe komponensek

| Komponens | Modell fájl | Kimenet |
|-----------|-------------|---------|
| `PoseLandmarker` | `assets/mediapipe/pose_landmarker.task` | 33 testpont (x, y, z, visibility) |
| `FaceLandmarker` | `assets/mediapipe/face_landmarker.task` | 468 arcpont (x, y, z) |
| `HandLandmarker` | `assets/mediapipe/hand_landmarker.task` | max. 2 kéz × 21 pont (x, y, z), bal/jobb címke |

Kód: `src/jelnyelv/mp_features.py` — `HolisticTasks`, `mediapipe_detection`, `pack_landmarks`, `scale_keypoint_vector`, `extract_keypoints`.

Felvétel / import: teljes felbontású képkocka (webkamera). Felismerés élőben: kép átméretezve `RECOGNITION_INPUT_WIDTH` (256 px szélesség) — gyorsabb, de ugyanaz a landmark kinyerés.

## Nyers vektor elrendezés (`pack_landmarks`)

Egy képkocka = **1662** float32 érték, **skála nélkül** mentve lemezre (`data/jelek/<szó>/<szekv>/<frame>.npy`).

| Blokk | Index (0-alapú) | Dimenzió | Tartalom |
|-------|-----------------|----------|----------|
| Test (pose) | 0 – 131 | 33×4 = **132** | x, y, z, visibility minden pose pontnál |
| Arc | 132 – 1535 | 468×3 = **1404** | x, y, z (arc mindig kitöltve, ha detektálva) |
| Bal kéz | 1536 – 1598 | 21×3 = **63** | x, y, z; ha nincs bal kéz → **nulla vektor** |
| Jobb kéz | 1599 – 1661 | 21×3 = **63** | x, y, z; ha nincs jobb kéz → **nulla vektor** |

Koordináták normalizált képtér (MediaPipe 0–1 tartomány), nem méterben.

### Egykezes jelek

- Csak **egy** kéz detektálása esetén a másik kéz blokkja **végig nulla** marad.
- Nincs külön „kéz látható” bit a vektorban — a modell a nulla blokkot implicit „nincs kéz” jelnek tanulja.
- Ha a detektor **nem látja** a jelet mutató kezet (sok nulla képkocka), a modell jobban támaszkodik arcra/testre → gyenge / összekevert osztályok.

### Hiányzó detekció

- Pose / arc / kéz hiányában a megfelelő blokk **nullázva** marad ( inicializált `zeros` ).

## Skálázás (tanítás + élő felismerés)

A lemezen **nyers** vektor van; skálázás:

- betöltéskor: `dataset.py` → `scale_keypoint_vector`
- élőben: `extract_keypoints` → `scale_keypoint_vector`

Konstansok: `src/jelnyelv/config.py`

| Blokk | Skála (jelenlegi) | Cél |
|-------|-------------------|-----|
| `POSE_FEATURE_SCALE` | **0.9** | test kicsit visszább |
| `FACE_FEATURE_SCALE` | **0.18** | arc sok dimenzió miatt erősen lefojtva |
| `HAND_FEATURE_SCALE` | **6.0** | bal + jobb kéz erősen kiemelve |

**Effektív „súly”** (durva becslés: dimenziószám × skála², hasonló koordináta-varians mellett):

- arc ~2–3%
- test ~2–3%
- két kéz együtt ~**94–96%**

Checkpointban is mentve: `config.pose/face/hand_feature_scale` a `your_model.pth`-ban.

**Skála változtatás után kötelező újratanítás** — a régi modell más súlyokkal lett tanítva.

## Szekvencia és tanító adat

| Paraméter | Érték |
|-----------|--------|
| Képkocka / szekvencia | `SEQUENCE_LENGTH` = 31 |
| Felvett szekvenciák / szó | `NO_SEQUENCES` = 30 |
| Bemenet az LSTM-nek | (31, 1662) skálázott vektor |
| Címkék | `data/jelek/` mappanevek |

## Gyakori hibák → mit nézzünk

| Tünet | Lehetséges ok |
|--------|----------------|
| Új szó 0% pontosság | Gesztus hasonló másik szóhoz landmark térben; vagy sok **nulla kéz** képkocka |
| apa ↔ köszönöm ↔ kutya keveredés | Hasonló kézpozíció; arc/test még mindig differenciál |
| Egykezes jel instabil | Másik kéz néha félrement detektálva; vagy kez oldala vált |
| Régi modell + új skála | Nem kompatibilis — újra `train` |

### Gyors ellenőrzés (nulla kéz aránya egy szekvenciában)

```bash
. .venv/bin/activate
python -c "
import os, numpy as np
from jelnyelv.config import DATA_PATH, POSE_DIMS, FACE_DIMS, HAND_DIMS, SEQUENCE_LENGTH
hs = POSE_DIMS + FACE_DIMS
word, seq = 'kutya', '0'
for f in range(SEQUENCE_LENGTH):
    v = np.load(os.path.join(DATA_PATH, word, seq, f'{f}.npy'))
    zl = np.allclose(v[hs:hs+HAND_DIMS], 0)
    zr = np.allclose(v[hs+HAND_DIMS:hs+2*HAND_DIMS], 0)
    if zl or zr: print(f'frame {f}: bal_null={zl} jobb_null={zr}')
"
```

## Kapcsolódó fájlok

- `src/jelnyelv/config.py` — skálák, útvonalak
- `src/jelnyelv/mp_features.py` — landmark kinyerés és pakolás
- `src/jelnyelv/dataset.py` — betöltés + skálázás tanításhoz
- `ertekelesi_jelentes.txt` / `ertekelesi_confusion_matrix.png` — utolsó kiértékelés

Utolsó skála-beállítás dokumentálva: **kéz-hangú** (`FACE=0.18`, `HAND=6.0`, `POSE=0.9`).
