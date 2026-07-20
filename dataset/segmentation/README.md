Распакуйте сюда содержимое ZIP «YOLO Seg» с сайта. В корне должны оказаться `data.yaml`, `images/train`, `images/val`, `labels/train` и `labels/val`. Папка `test` не используется.

Можно не распаковывать архив вручную:

```powershell
python tools/segmentation/train.py --data "C:\путь\к\yolo_segment.zip"
```
