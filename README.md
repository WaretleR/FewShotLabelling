# FewShotLabelling

Данный скрипт представляет собой прототип системы автоматизации разметки видео для задачи распознавания действий людей. Прототип принимаем на вход любой набор видео, организованный в формате датасета [VIRAT](https://viratdata.org/), а также произвольное количество разметки, также в формате [VIRAT](https://viratdata.org/). Данная разметка может быть получена, к примеру, ручной разметкой в одном из существующих средств ручной разметки(к примеру, [CVAT](https://github.com/openvinotoolkit/cvat)).

## Формат данных VIRAT
### Структура датасета

```
dataset
└───videos
│   |   video1.mp4
│   |   video2.mp4
│   |   ...
│   
└───labels
    │   video1.viratdata.events.txt
    │   video2.viratdata.events.txt
    |   ...
```
### Формат файла разметки

| ID | Класс | Длительность (в кадрах) | Первый кадр | Последний кадр | Текущий кадр | xmin | ymin | xlen | ylen |
| --:| -----:| -----------------------:| -----------:| --------------:| ------------:| ----:| ----:| ----:| ----:|
|  14|      7|                       63|          400|             462|           400|   277|   273|    70|   130|
| ...|    ...|                      ...|          ...|             ...|           ...|   ...|   ...|   ...|   ...|

ID событий должны быть уникальны в рамках одного видео (т.е. в рамках одного файла разметки).

## Запуск прототипа

```console
python prototype.py \
  -labels /path/to/labels \
  -data /path/to/video \
  -target /path/to/output \
  -selected /path/to/file/with/video/names \
  -classesNames /path/to/file/with/classes/names \
  -frameFreqMOT 8 \
  -premadeTracks /path/to/saved/tracks
```

<b>-labels</b> - путь к созданной вручную разметке

<b>-data</b> - путь к набору видео

<b>-target</b> - путь для сохранения видео и разметки

<b>-selected</b> - видео из data, для которых надо строить разметку (если не указано, то строится для всех); файл должен содержать названия файлов видео (по 1 на строке)

<b>-classesNames</b> - названия классов (если не указано, будут использоваться их номера из разметки); файл должен содержать названия всех классов (по 1 на строке)

<b>-frameFreqMOT</b> - через сколько кадров обновлять треки объектов в видео (по умолчанию - 8)

<b>-premadeTracks</b> - если указано, то используются сгенерированные ранее треки (по умолчанию генерируются новые); можно использовать для визуализации ground truth

## Структура выходной директории

```
targetDir
└───videos
│   |   video1.avi
│   |   video2.avi
│   |   ...
│   
└───labels
    │   video1.viratdata.events.txt
    │   video2.viratdata.events.txt
    |   ...
```

Видео содержат bbox вокруг всех обнаруженных людей во всех кадрах. Если были указаны имена классов, то их классы действий подписаны над bbox.

Разметка содержит описания всех найденных и классифицированных треков в формате VIRAT.
