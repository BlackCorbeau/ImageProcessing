import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

# Загрузка изображения
image = cv2.imread('sar_3.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Часть 1: Нахождение наиболее протяженного участка с помощью преобразования Хафа
print("=== Часть 1: Поиск наиболее протяженного участка (преобразование Хафа) ===")

# Предобработка для улучшения детекции линий
blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# Применение преобразования Хафа для линий
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Поиск самой длинной линии
longest_line = None
max_length = 0

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > max_length:
            max_length = length
            longest_line = line[0]

# Визуализация результатов
plt.figure(figsize=(15, 10))

# Исходное изображение
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение')
plt.axis('off')

# Границы
plt.subplot(2, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title('Границы (Canny)')
plt.axis('off')

# Все линии Хафа
image_with_lines = image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
plt.title(f'Все линии Хафа ({len(lines)} найдено)')
plt.axis('off')

# Самая длинная линия
image_longest_line = image.copy()
if longest_line is not None:
    x1, y1, x2, y2 = longest_line
    cv2.line(image_longest_line, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(image_longest_line, f'Length: {max_length:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(image_longest_line, cv2.COLOR_BGR2RGB))
plt.title('Самая длинная линия')
plt.axis('off')

# Часть 2: Исследование алгоритмов бинаризации для выделения дорожной полосы
print("\n=== Часть 2: Исследование алгоритмов бинаризации ===")

# Различные методы бинаризации
methods = {
    'Global Otsu': cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU),
    'Global Binary': cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY),
    'Adaptive Mean': cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 11, 2),
    'Adaptive Gaussian': cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
}

# Применение морфологических операций для улучшения результатов
kernel = np.ones((3, 3), np.uint8)

plt.subplot(2, 3, 5)
for i, (name, result) in enumerate(methods.items()):
    if len(result) == 2:  # Для глобальных методов
        _, binary = result
    else:  # Для адаптивных методов
        binary = result
    
    # Морфологические операции
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel)
    
    # Поиск контуров дорожной полосы
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Фильтрация контуров по площади
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    
    # Визуализация для Otsu (обычно лучший результат)
    if name == 'Global Otsu':
        image_road = image.copy()
        cv2.drawContours(image_road, large_contours, -1, (0, 255, 0), 3)
        
        plt.imshow(cv2.cvtColor(image_road, cv2.COLOR_BGR2RGB))
        plt.title(f'Дорожная полоса (Otsu)\nКонтуров: {len(large_contours)}')
        plt.axis('off')

# Сравнение всех методов бинаризации
plt.subplot(2, 3, 6)
comparison_images = []
titles = []

for name, result in methods.items():
    if len(result) == 2:
        _, binary = result
    else:
        binary = result
    comparison_images.append(binary)
    titles.append(name)

# Создание коллажа для сравнения
collage = np.vstack([
    np.hstack(comparison_images[:2]),
    np.hstack(comparison_images[2:])
])

plt.imshow(collage, cmap='gray')
plt.title('Сравнение методов бинаризации')
plt.axis('off')

plt.tight_layout()
plt.show()

# Детальный анализ для лучшего метода (Otsu)
print("\nДетальный анализ метода Otsu:")
_, otsu_binary = methods['Global Otsu']

# Улучшение бинаризации
otsu_clean = cv2.morphologyEx(otsu_binary, cv2.MORPH_OPEN, kernel)
otsu_clean = cv2.morphologyEx(otsu_clean, cv2.MORPH_CLOSE, kernel)

# Поиск и анализ контуров
contours, _ = cv2.findContours(otsu_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

print(f"Всего контуров: {len(contours)}")
print(f"Крупных контуров (площадь > 500): {len(large_contours)}")

# Выделение самого большого контура (предположительно дорожной полосы)
if large_contours:
    largest_contour = max(large_contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    print(f"Площадь самого большого контура: {area:.1f}")
    
    # Создание маски для самого большого контура
    mask = np.zeros_like(image_gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    # Применение маски к исходному изображению
    road_extracted = cv2.bitwise_and(image, image, mask=mask)
    
    # Визуализация результата
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Маска дорожной полосы')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(road_extracted, cv2.COLOR_BGR2RGB))
    plt.title('Выделенная дорожная полоса')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Результаты преобразования Хафа
if longest_line is not None:
    x1, y1, x2, y2 = longest_line
    print(f"\nРезультаты преобразования Хафа:")
    print(f"Координаты самой длинной линии: ({x1}, {y1}) - ({x2}, {y2})")
    print(f"Длина линии: {max_length:.1f} пикселей")
    print(f"Угол наклона: {np.arctan2(y2-y1, x2-x1) * 180/np.pi:.1f} градусов")
