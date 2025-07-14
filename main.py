import math
import time
import pygetwindow as gw
import pyautogui as pg
import cv2
import numpy as np
import pygame
import start_window
import setting
import math
import copy
# Настройка Pygame
pygame.init()

# Название окна игры
GAME_WINDOW_TITLE = "BlueStacks App Player"

# Координаты углов игрового поля
FIELD_TOP_LEFT = setting.FIELD_TOP_LEFT
FIELD_TOP_RIGHT = setting.FIELD_TOP_LEFT
FIELD_BOTTOM_RIGHT = setting.FIELD_TOP_LEFT
FIELD_BOTTOM_LEFT = setting.FIELD_TOP_LEFT

# Вычисленные размеры поля
FIELD_WIDTH = setting.FIELD_WIDTH
FIELD_HEIGHT = setting.FIELD_HEIGHT

# Параметры объектов
BALL_RADIUS = setting.BALL_RADIUS # радиус мяча
DISC_RADIUS = setting.DISC_RADIUS  # Радиус фишек (~26 пикселей)
GOAL_WIDTH = setting.GOAL_WIDTH  # Ширина ворот
GOAL_DEPTH = setting.GOAL_DEPTH  # Глубина ворот

def window():
    hwnd = start_window.main()  # Ищем окно и закрепляем его
    if not hwnd:
        return
    return hwnd

HWND = window()


def correct_visual_coords(coords):
    return (coords[0] + 6, coords[1])

def absolute_to_image_players_coords(coords):
    return (coords[0] - FIELD_TOP_LEFT[0], coords[1] - FIELD_TOP_LEFT[1] + DISC_RADIUS)

def absolute_to_image_ball_coords(coords):
    return (coords[0]- FIELD_TOP_LEFT[0],coords[1]- FIELD_TOP_LEFT[1] + 2*BALL_RADIUS)

def image_players_to_absolute_coords(coords):
    return (coords[0] + FIELD_TOP_LEFT[0], coords[1] + FIELD_TOP_LEFT[1] - DISC_RADIUS)

def image_ball_to_absolute_coords(coords):

    return (coords[0] +  FIELD_TOP_LEFT[0],coords[1]+ FIELD_TOP_LEFT[1] - 2*BALL_RADIUS )

def capture_screenshot_players(window = HWND):
    """Делает скриншот области игрового поля."""
    if window is None:
        return None
    x, y = FIELD_TOP_LEFT
    width, height = FIELD_WIDTH, FIELD_HEIGHT
    screenshot = pg.screenshot(region=(x, y-DISC_RADIUS, width, height + 2* DISC_RADIUS))
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    return screenshot

def capture_screenshot_ball(window = HWND):
    """Делает скриншот области игрового поля."""
    if window is None:
        return None
    x, y = FIELD_TOP_LEFT
    width, height = FIELD_WIDTH, FIELD_HEIGHT
    screenshot = pg.screenshot(region=(x, y - 2*BALL_RADIUS, width, height + 4* BALL_RADIUS))
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    return screenshot


def process_image(image, threshold = 40):
    """Делает фон белым, а фишки черными."""

    # Переводим в цветовое пространство HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # Порог насыщенности (чтобы отделить серый фон от цветных объектов)
    threshold = threshold

    # Если насыщенность ниже порога — фон (делаем белым)
    # Если насыщенность выше — считаем цветными объектами (делаем черным)
    mask = (s <= threshold).astype(np.uint8) * 255

    # Инвертируем, чтобы фон стал белым, а объекты — черными
    final_image = cv2.bitwise_not(mask)


    return final_image



def find_players():

    image = chernkovik('players')
    # Применяем Hough Transform для поиска кругов
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=13, minRadius=DISC_RADIUS-5,
                               maxRadius=DISC_RADIUS )

    players_coord = []
    # Если круги найдены, рисуем их на изображении
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Преобразуем изображение в цветное для рисования
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Рисуем круги
        for (x, y, r) in circles:
            # Рисуем круги на изображении (красным цветом)
            cv2.circle(color_image, (x, y), r, (0, 0, 255), 2)  # Красный цвет, толщина 2
            players_coord.append((x,y))

        if False:
            # Показываем результат
            cv2.imshow("Circles Detected", color_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return np.array(players_coord)
def find_ball():

    image = chernkovik('ball')


    # Инвертируем, чтобы черное стало белым
    inverted = cv2.bitwise_not(image)

    # Размываем, чтобы убрать шум и мелкие дефекты
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)

    # Ищем круги методом Хафа
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=5,
                               param1=50, param2=5, minRadius=BALL_RADIUS-1, maxRadius=BALL_RADIUS+1)

    # Преобразуем в цветное изображение для отрисовки
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    best_circle = None
    best_black_ratio = 0  # Максимальный процент чёрных пикселей

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for x, y, r in circles:
            # Вырезаем область внутри круга
            mask = np.zeros_like(image)
            cv2.circle(mask, (x, y), r, 255, -1)  # Белая маска круга
            ball_region = cv2.bitwise_and(image, image, mask=mask)

            # Считаем количество чёрных пикселей внутри круга
            total_pixels = np.count_nonzero(mask)  # Все пиксели в круге
            black_pixels = np.count_nonzero(ball_region > 200)  # Чёрные пиксели (порог 50)

            black_ratio = black_pixels / total_pixels if total_pixels > 0 else 0

            if black_ratio > best_black_ratio:
                best_black_ratio = black_ratio
                best_circle = (x, y, r)

        if best_circle:
            x, y, r = best_circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)

    else:
        print("Мяч не найден")

    if False:
        # Показываем изображение
        cv2.imshow("Detected Circles", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return np.array([x, y])


def find_mine_command(coordinates):
    """
    Для каждой координаты в списке `coordinates` находит среднее значение пикселей
    в области радиуса `radius` вокруг каждой точки.

    :param image: Исходное изображение (cv2.imread() или результат обработки).
    :param coordinates: Список координат (x, y), для которых нужно вычислить среднее значение пикселей.
    :param radius: Радиус области для усреднения.

    :return: Список средних значений пикселей для каждой координаты.
    """

    image = capture_screenshot_players()
    averages = []

    # Для каждого пикселя из списка координат

    data = []
    for (x, y) in coordinates:
        r = DISC_RADIUS
        # Вычисляем границы области с учетом радиуса
        x_start = max(0, x - r)
        x_end = min(image.shape[1], x + r)
        y_start = max(0, y - r)
        y_end = min(image.shape[0], y + r)

        # Извлекаем область изображения
        region = image[y_start:y_end, x_start:x_end]

        # Создаем маску для круга
        yy, xx = np.ogrid[y_start:y_end, x_start:x_end]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2

        # Применяем маску к изображению
        region_masked = region[mask]

        # Вычисляем средний цвет только внутри круга
        mean_value = np.mean(region_masked, axis=0)
        color = mean_value
        color = (color[2], color[1], color[0])

        #Все выше не используем, проверяем только один-два пикселя
        color = image[y][x]

        if (20 < color[0] and color[0] < 70) and (10 < color[1] and color[1] < 30) and (200 < color[2] and color[2] < 260):
            my_command = True
        else:
            my_command = False

        pos = (int(x),int(y))

        data.append([pos, my_command, color])


    return data

def chernkovik(mode):
    # Загружаем изображение
    if mode == 'ball':
        image = capture_screenshot_ball()
    elif mode == 'players':
        image = capture_screenshot_players()


    def AND(filter1, filter2):

        filter1 = NOT(filter1)
        filter2 = NOT(filter2)
        """
        Объединяет два фильтра методом И:
        Пиксель остаётся черным только если он черный в обеих масках.
        """
        combined = cv2.bitwise_and(filter1, filter2)
        combined = NOT(combined)

        return combined

    def OR(filter1, filter2):
        return cv2.bitwise_and(filter1, filter2)

    def NOT(filter):
        return cv2.bitwise_not(filter)

    def f1(image, brightness_threshold=220):
        """
        Фильтрует изображение по яркости.
        Пиксели с яркостью меньше brightness_threshold становятся белыми,
        а остальные черными.
        """
        # Преобразуем изображение в градации серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Пиксели с яркостью меньше threshold становятся белыми (255), остальные черными (0)
        _, binary_filter = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY_INV)

        return binary_filter

    def f2(image):
        """
           Фильтрует изображение: если каналы пикселя одинаковы (пиксель серый), он будет черным,
           если каналы различаются (пиксель цветной), он будет белым.
           """
        # Разделяем изображение на каналы
        (b, g, r) = cv2.split(image)

        # Вычисляем абсолютное различие между каналами
        diff_rg = cv2.absdiff(r, g)
        diff_rb = cv2.absdiff(r, b)
        diff_gb = cv2.absdiff(g, b)

        # Применяем порог, чтобы выделить пиксели с маленьким различием между каналами
        _, mask_rg = cv2.threshold(diff_rg, 15, 255, cv2.THRESH_BINARY)
        _, mask_rb = cv2.threshold(diff_rb, 15, 255, cv2.THRESH_BINARY)
        _, mask_gb = cv2.threshold(diff_gb, 15, 255, cv2.THRESH_BINARY)

        # Объединяем маски, используя логическое "И"
        final_mask = cv2.bitwise_and(mask_rg, mask_rb)
        final_mask = cv2.bitwise_and(final_mask, mask_gb)

        # Если каналы отличаются, пиксель будет белым, иначе черным
        result = np.zeros(image.shape[:2], dtype=np.uint8)
        result[final_mask == 255] = 255  # Белые пиксели для цветных, черные для серых

        return result

    def f3(image, threshold):
        return process_image(image,threshold)

    def f4(image):
        """
            Преобразует изображение в черно-белое, где:
            - Пиксели с яркостью меньше 30 или больше 200 становятся черными.
            - Остальные пиксели становятся белыми.
            """
        # Преобразуем изображение в оттенки серого
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Создаем пустую маску для результата
        result = np.zeros(gray_image.shape, dtype=np.uint8)

        # Применяем условие для яркости пикселей |||||
        result[(gray_image < 100) | (gray_image > 255)] = 0  # Черные пиксели
        result[(gray_image >= 100) & (gray_image <= 255)] = 255  # Белые пиксели

        return result



    # Применяем фильтрации
    filter1 = f1(image)
    filter2 = f2(image)
    filter4 = f4(image)

    ball = NOT(AND(f3(image,100),f3(image,30)))

    players =  (OR(f3(image,50),f3(image,30)))

    # Загружаем изображение
    if mode == 'ball':
        return ball
    elif mode == 'players':
        return players




def choice_coord(coord_piece):
    image_piece_coord = absolute_to_image_players_coords(coord_piece)
    def filter_yellow_arrow(image):

        """Фильтр для выделения светло-желтого цвета"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([23, 160, 230])  # Нижняя граница
        upper_yellow = np.array([25, 180, 255])  # Верхняя граница
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        return mask

    def find_arrow_tip(mask):
        """Находит самую удаленную точку контура относительно заданной"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return coord_piece  # Если контуров нет, вернуть None

        # Найти самый большой контур (стрелка)
        largest_contour = max(contours, key=cv2.contourArea)

        # Начальная точка наконечника
        x0, y0 = image_piece_coord

        # Найти точку, наиболее удалённую от (x0, y0)
        farthest_point = max(largest_contour, key=lambda point: (point[0][0] - x0) ** 2 + (point[0][1] - y0) ** 2)

        return farthest_point[0]  # Координаты наконечника (x, y)

    image = capture_screenshot_players()

    # Фильтруем желтый цвет
    mask = filter_yellow_arrow(image)


    # Находим новый наконечник стрелки
    arrow_tip = find_arrow_tip(mask)


    # Отображение результата
    if False:
        cv2.circle(image, arrow_tip, 5, (0, 0, 255))  # Отмечаем наконечник красным
        cv2.circle(image, image_piece_coord, 5, (0, 255, 255))

        cv2.imshow("Arrow Ti", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Arrow Tip", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    arrow_tip = image_players_to_absolute_coords(arrow_tip)
    arrow_tip = (int(arrow_tip[0]),int(arrow_tip[1]))
    return arrow_tip

window()
