import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import requests


def _get_grid(grid_shape):
    """
    Функция генерации сетки по текущему уровню
    """
    grid = []  # Массив для финальной сетки
    grid_row = []  # Массив для столбца
    for i in range(grid_shape[0]):  # По всем строкам
        for j in range(grid_shape[1]):  # По всем столбцам
            grid_row.append([j, i])  # Создаем элемент [j, i]
        grid.append(grid_row)  # Добавляем столбец в финальную сетку
        grid_row = []  # Обнуляем данные для столбца
    grid = np.array(grid)  # Переводим в numpy
    grid = np.expand_dims(grid, axis=2)  # Добавляем размерность
    return grid


# Функция расчета сигмоиды для вектора
def sigmoid(x):  # На вход подаем массив данных
    return 1 / (1 + np.exp(-x))  # Возвращаем сигмоиду для всех элементов массива


def non_max_suppression_fast(boxes, scores, classes, overlapThresh):
    if len(boxes) == 0:  # Если нет ни одного бокса
        return []

    pick = []  # Индексы возвращаемых боксов

    x1 = boxes[:, 0]  # координаты x левыъ верхних углов
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int"), scores[pick], classes[pick].astype("int")


class YOLOProcess:
    def __init__(self,
                 input_shape=(416, 416, 3), num_sub_anchors=9 // 3,
                 name_classes=[''],
                 colors=['green'],
                 anchors=np.array(
                     [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198],
                      [373, 326]]),
                 anchor_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 font_file_name='font.otf'):
        self.input_shape = np.array([input_shape[0], input_shape[1]])
        self.anchors = anchors
        self.anchor_mask = anchor_mask
        self.num_anchors = len(anchors)
        self._boxes = {}
        self._box_scores = {}
        self._boxes_out = {}
        self._scores_out = {}
        self._classes_out = {}
        self.name_classes = name_classes
        self.num_classes = len(name_classes)
        self.font_file_name = font_file_name
        self.colors = colors
        self.layers = ['conv2d_58', 'conv2d_66', 'conv2d_74']

    def _load_img(self, file_name):
        """
        Функция загрузки картинки.
        Картинка сохраняется в self.img
        Одновременно с картинкой загружается шрифт для надписи
        """
        self.image_file_name = file_name
        self.img = Image.open(self.image_file_name)  # Загружаем изображение
        self.image_shape = np.array([self.img.size[1], self.img.size[0]])  # Сохраняем размер оригинального изображения
        self.img_for_predict = self.img.resize(self.input_shape, Image.BICUBIC)
        self.img_for_predict = np.array(self.img_for_predict) / 255.
        self.img_for_predict = self.img_for_predict.reshape(1, self.input_shape[0], self.input_shape[1], 3)

        self.font = ImageFont.truetype(font=self.font_file_name,
                                       size=np.floor(3e-2 * self.img.size[1] + 0.5).astype('int32'))
        self.thickness = (self.img.size[0] + self.img.size[1]) // 300

    def _predict(self):
        """
        Функция вызывает predict модели и решейпит результат
        """

        data = json.dumps({"signature_name": '', "inputs": self.img_for_predict.tolist()})
        headers = {"content-type": "application/json"}
        response = requests.post('http://localhost:8501/v1/models/fishdetect_model:predict', data=data,
                                 headers=headers)

        result = response.json()["outputs"]
        self.predicts = []

        for key in self.layers:
            p = np.array(result[key])
            v = p.reshape(p.shape[0], p.shape[1], p.shape[2], 3, -1)
            self.predicts.append(v)

    def _find(self, lvl):
        """
              Функция проходит по ответу полученному от модели и переводит все боксы
              в абсолютные координаты картинки
            """
        predict = self.predicts[lvl]
        grid_shape = predict.shape[1:3]

        grid = _get_grid(grid_shape)
        num_anchors = len(self.anchors[self.anchor_mask[lvl]])
        anchors_tensor = np.reshape(self.anchors[self.anchor_mask[lvl]], (1, 1, 1, num_anchors, 2))

        # Получаем параметры бокса

        # Координаты центра bounding box
        xy_param = predict[...,
                   :2]  # Берем 0 и 1 параметры из предикта (соответствуют параметрам смещения центра анкора)
        box_xy = (sigmoid(xy_param) + grid) / grid_shape[::-1]  # Получаем координаты центра bounding box

        # Высота и ширна bounding box
        wh_param = predict[...,
                   2:4]  # Берем 2 и 3 параметры из предикта (соответствуют праметрам изменения высоты и ширины анкора)
        box_wh = np.exp(wh_param) * anchors_tensor / self.input_shape[::-1]  # Получаем высоту и ширину bounding box

        # Вероятность наличия объекта в анкоре
        conf_param = predict[...,
                     4:5]  # Берем 4 параметр из предикта (соответствуют вероятности обнаружения объекта)
        box_confidence = sigmoid(conf_param)  # Получаем вероятность наличия объекта в bounding box

        # Класс объекта
        class_param = predict[...,
                      5:]  # Берем 5+ параметры из предикта (соответствуют вероятностям классов объектов)
        box_class_probs = sigmoid(class_param)  # Получаем вероятности классов объектов

        # Корректируем ограничивающие рамки (Размер изображения на выходе 416х416)
        # И найденные параметры соответствуют именно этой размерности
        # Необходимо найти координаты bounding box для рамерности исходного изображения
        box_yx = box_xy[..., ::-1].copy()
        box_hw = box_wh[..., ::-1].copy()

        new_shape = np.round(self.image_shape * np.min(
            self.input_shape / self.image_shape))  # Находим размерность пропорциональную исходной с одной из сторон 416
        offset = (
                         self.input_shape - new_shape) / 2. / self.input_shape  # Смотрим на сколько надо сместить в относительных координатах
        scale = self.input_shape / new_shape  # Находим коэфициент масштабирования
        box_yx = (box_yx - offset) * scale  # Смещаем по координатам
        box_hw *= scale  # Масштабируем ширину и высоту

        box_mins = box_yx - (
                box_hw / 2.)  # Получаем левые верхние координаты (от середины отнимаем половину ширины и высоты)
        box_maxes = box_yx + (
                box_hw / 2.)  # Получаем правые нижнние координаты (к середине прибавляем половину ширины и высоты)
        boxes = np.concatenate([
            box_mins[..., 0:1],  # yMin
            box_mins[..., 1:2],  # xMin
            box_maxes[..., 0:1],  # yMax
            box_maxes[..., 1:2]  # xMax
        ], axis=-1)

        boxes *= np.concatenate(
            [self.image_shape, self.image_shape])  # Переводим из относительных координат в абсолютные
        self._boxes[lvl] = boxes
        self._box_scores[lvl] = box_confidence * box_class_probs

    def draw_boxes(self, _classes_out, _boxes_out, _scores_out):
        """
        рисует боксы на картинке сохраненной в
        self.image_pred
        """
        for i, c in reversed(list(enumerate(_classes_out))):
            draw = ImageDraw.Draw(self.image_pred)
            predicted_class = self.name_classes[c]
            box = _boxes_out[i]
            score = _scores_out[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            label_size = draw.textsize(label, self.font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(self.img.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(self.img.size[0], np.floor(right + 0.5).astype('int32'))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(self.thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=self.font)
            del draw

    def process(self, file_name):
        """
        Загружает картинку, предиктит и извлекает из предикта все боксы по всем уровням

        """

        self._load_img(file_name)
        self._predict()

        for lvl, _ in enumerate(self.anchor_mask):
            self._find(lvl)
            boxes_reshape = np.reshape(self._boxes[lvl], (-1, 4))  # Решейпим все боксы в один массив
            box_scores_reshape = np.reshape(self._box_scores[lvl], (-1, self.num_classes))  # Решейпим в один массив

            # Что бы не хранить много боксов для дальнейшей обработки
            # возьмем боксы в которых что-то есть с вероятностью больше 1%

            mask = np.max(box_scores_reshape, axis=1) > 0.01

            self._boxes_out[lvl] = boxes_reshape[mask]
            self._scores_out[lvl] = np.max(box_scores_reshape, axis=1)[mask]
            self._classes_out[lvl] = np.argmax(box_scores_reshape, axis=1)[mask]

    def applay_mask(self, result_confidence=0.7):
        """
        Отрисовывает боксы вероятность нахождения объекта в которых выше заданной.
        """

        self.image_pred = self.img.copy()
        new_boxes = []
        new_scores = []
        new_classes = []
        for lvl, _ in enumerate(self.anchor_mask):
            if len(self._boxes_out[lvl]) > 0:
                new_boxes += list(self._boxes_out[lvl])
                new_scores += list(self._scores_out[lvl])
                new_classes += list(self._classes_out[lvl])

        new_scores = np.array(new_scores)
        new_boxes = np.array(new_boxes)
        new_classes = np.array(new_classes)
        # Берем все объекты, обнаруженные с заданной вероятностью
        mask = new_scores >= result_confidence

        new_scores = new_scores[mask]
        new_boxes = new_boxes[mask]
        new_classes = new_classes[mask]

        self.draw_boxes(new_classes, new_boxes, new_scores)

        return self.image_pred

    def applay_max_suppression_fast_filter(self, result_confidence=0.7, coincidence_param=0.15):

        self.image_pred = self.img.copy()
        for lvl, _ in enumerate(self.anchor_mask):
            mask = self._scores_out[lvl] >= result_confidence
            new_boxes = self._boxes_out[lvl][mask]
            new_scores = self._scores_out[lvl][mask]
            new_classes = self._classes_out[lvl][mask]

            if len(new_boxes) > 0:
                new_boxes, new_scores, new_classes = non_max_suppression_fast(new_boxes, new_scores, new_classes,
                                                                              coincidence_param)
                self.draw_boxes(new_classes, new_boxes, new_scores)

        return self.image_pred

    def applay_max_filter(self):
        """
              Отрисовывает бокс с максимальной вероятностью нахождения объекта.
              функцию удобно использовать при отладке
            """

        self.image_pred = self.img.copy()
        new_boxes = []
        new_scores = []
        new_classes = []
        for lvl, _ in enumerate(self.anchor_mask):
            if len(self._boxes_out[lvl]) > 0:
                new_boxes += list(self._boxes_out[lvl])
                new_scores += list(self._scores_out[lvl])
                new_classes += list(self._classes_out[lvl])

        # Берем все объекты, обнаруженные с максимальной вероятностью
        mask = new_scores == max(new_scores)
        new_boxes = np.array(new_boxes)[mask]
        new_scores = np.array(new_scores)[mask]
        new_classes = np.array(new_classes)[mask]

        self.draw_boxes(new_classes, new_boxes, new_scores)

        return self.image_pred


if __name__ == "__main__":
    pass
