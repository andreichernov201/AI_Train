Распакуйте сюда содержимое ZIP «YOLO Detect» с сайта. В корне должны оказаться `data.yaml`, `images/train`, `images/val`, `labels/train` и `labels/val`. Папка `test` не используется.

Можно не распаковывать архив вручную:

```powershell
python tools/detect/train.py --data "C:\путь\к\yolo_detect.zip"
```
