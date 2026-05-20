# training_photos (algemene leer-map + Roboflow ZIP)

Ja: je kan **meerdere databases/ZIPs** in 1 map zetten.

## Ondersteund
- Alle `.zip` bestanden in:
  - `training_photos/roboflow_zip/`
  - `C:\Users\marie\Downloads\yolov8`
  - extra paden via `ROBOFLOW_ZIP_EXTRA_DIRS`
- Scannen gebeurt **recursief** (dus ook submappen in die map).

## Werking
1. Zet 1 of meerdere YOLOv8/Roboflow ZIPs in die map(pen).
2. Start `face_tracker.py`.
3. Script extract automatisch elke ZIP naar:
   - `training_photos/roboflow_extracted/<zip_naam>/`
4. Alle afbeeldingen uit al die extracties worden gebruikt voor het leerprofiel.

Dus ja: meer ZIPs = meer data = meestal stabielere tracking.


## Servo stopt op X30/Y30?
Als je servo firmware groter bereik heeft, zet:
- `SERVO_X_MIN=0`
- `SERVO_X_MAX=180`
- `SERVO_Y_MIN=0`
- `SERVO_Y_MAX=180`

Script gebruikt nu standaard 0..180 en toont limieten in beeld.
