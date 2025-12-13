# Инструкция по установке датсета

Датасет на котором обучались модели находится по ссылке: https://www.kaggle.com/datasets/radmilasegen/dataset-fin

Для его установки используйте следующие команды:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("radmilasegen/dataset-fin")

print("Path to dataset files:", path)
```

В файле [merged_classes.txt](merged_classes.txt) представлен список детектируемых объектов. Этот же файл можно найти в датасете.
