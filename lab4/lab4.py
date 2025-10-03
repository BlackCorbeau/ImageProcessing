import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import segmentation, feature, filters, measure, morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Загрузка изображения (предполагаем, что есть изображение с газоном и пальмами)
# Для демонстрации создадим тестовое изображение или загрузим существующее
try:
    image = cv2.imread('lawn_image.jpg')
    if image is None:
        raise FileNotFoundError
    print(f"Загружено изображение размером: {image.shape}")
except:
    # Создаем тестовое изображение для демонстрации
    print("Создаем тестовое изображение...")
    image = np.ones((400, 600, 3), dtype=np.uint8) * 100
    # Газон (зеленая область)
    image[100:300, 100:500] = [50, 150, 50]
    # Пальмы (коричневые стволы и зеленые кроны)
    for i in range(5):
        x = 120 + i * 100
        # Ствол
        image[150:250, x-5:x+5] = [80, 50, 20]
        # Крона
        cv2.circle(image, (x, 120), 30, (40, 120, 40), -1)
    print(f"Создано тестовое изображение размером: {image.shape}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(f"Размер изображения: {image_rgb.shape}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Исходное изображение')
plt.axis('off')

# Часть 1: Алгоритм разрастания регионов для выделения газона
print("=== Часть 1: Алгоритм разрастания регионов для выделения газона ===")

class RegionGrowing:
    def __init__(self):
        self.seeds = []
        self.region = None
        
    def homogeneity_criterion_color(self, pixel, region_mean, threshold=20, region_pixels=None):
        """Критерий однородности на основе цвета (евклидово расстояние в RGB)"""
        return np.linalg.norm(pixel - region_mean) < threshold
    
    def homogeneity_criterion_intensity(self, pixel, region_mean, threshold=15, region_pixels=None):
        """Критерий однородности на основе интенсивности"""
        return abs(pixel - region_mean) < threshold
    
    def homogeneity_criterion_variance(self, pixel, region_mean, region_pixels, threshold=25):
        """Критерий однородности на основе дисперсии региона"""
        if len(region_pixels) < 2:
            return True
        current_variance = np.var(region_pixels)
        temp_pixels = region_pixels + [pixel]
        new_variance = np.var(temp_pixels)
        return abs(new_variance - current_variance) < threshold
    
    def homogeneity_criterion_gradient(self, pixel, region_mean, threshold=10, region_pixels=None):
        """Критерий однородности на основе градиента"""
        if region_pixels is None or len(region_pixels) < 2:
            return True
        # Вычисляем градиент между текущим пикселем и средним значением региона
        gradient = np.abs(pixel - region_mean)
        return np.max(gradient) < threshold
    
    def grow_regions(self, image, seeds, homogeneity_func, threshold, use_color=True):
        """Алгоритм разрастания регионов"""
        if use_color:
            h, w, c = image.shape
            visited = np.zeros((h, w), dtype=bool)
            region_mask = np.zeros((h, w), dtype=bool)
        else:
            h, w = image.shape
            visited = np.zeros((h, w), dtype=bool)
            region_mask = np.zeros((h, w), dtype=bool)
        
        regions = []
        
        for seed in seeds:
            # Проверяем корректность координат семени
            x, y = seed
            if x >= w or y >= h or x < 0 or y < 0:
                print(f"Пропускаем некорректное семя: {seed}")
                continue
                
            if visited[y, x]:
                continue
                
            region_pixels = []
            queue = [seed]
            region_mean = image[y, x]
            
            while queue:
                x, y = queue.pop(0)
                
                if visited[y, x]:
                    continue
                    
                current_pixel = image[y, x]
                
                # Вызываем критерий однородности с правильным количеством аргументов
                if homogeneity_func.__code__.co_argcount == 4:  # Для критерия дисперсии
                    is_homogeneous = homogeneity_func(current_pixel, region_mean, region_pixels, threshold)
                else:  # Для других критериев
                    is_homogeneous = homogeneity_func(current_pixel, region_mean, threshold, region_pixels)
                
                if is_homogeneous:
                    visited[y, x] = True
                    region_mask[y, x] = True
                    region_pixels.append(current_pixel)
                    
                    # Обновляем среднее значение региона
                    if use_color:
                        region_mean = np.mean(region_pixels, axis=0)
                    else:
                        region_mean = np.mean(region_pixels)
                    
                    # Добавляем соседей (8-связность)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                            queue.append((nx, ny))
            
            if np.any(region_mask):  # Добавляем только непустые регионы
                regions.append(region_mask.copy())
            region_mask.fill(False)  # Сбрасываем маску для следующего региона
            
        return regions

# Применяем алгоритм разрастания регионов
rg = RegionGrowing()

# Автоматически определяем семена на основе размера изображения
h, w = image_gray.shape
print(f"Размер изображения для семян: {w}x{h}")

# Выбираем семена для газона (в центре зеленых областей)
seeds = [
    (w//3, h//3),      # левый верхний квадрант
    (2*w//3, h//3),    # правый верхний квадрант  
    (w//3, 2*h//3),    # левый нижний квадрант
    (2*w//3, 2*h//3),  # правый нижний квадрант
    (w//2, h//2)       # центр
]

print(f"Используемые семена: {seeds}")

# Применяем с цветным критерием
print("Применяем алгоритм с цветным критерием...")
regions_color = rg.grow_regions(image_rgb, seeds, rg.homogeneity_criterion_color, threshold=35)

# Применяем с критерием интенсивности
print("Применяем алгоритм с критерием интенсивности...")
regions_intensity = rg.grow_regions(image_gray, seeds, rg.homogeneity_criterion_intensity, 
                                   threshold=30, use_color=False)

# Визуализация результатов
plt.subplot(1, 3, 2)
if regions_color:
    lawn_mask_color = np.any(regions_color, axis=0)
    image_with_lawn = image_rgb.copy()
    image_with_lawn[lawn_mask_color] = [255, 0, 0]  # Красным выделяем газон
    plt.imshow(image_with_lawn)
    plt.title(f'Выделение газона (цветной критерий)\nРегионов: {len(regions_color)}')
else:
    plt.imshow(image_rgb)
    plt.title('Не удалось выделить регионы (цветной критерий)')
plt.axis('off')

plt.subplot(1, 3, 3)
if regions_intensity:
    lawn_mask_intensity = np.any(regions_intensity, axis=0)
    image_with_lawn_intensity = image_rgb.copy()
    image_with_lawn_intensity[lawn_mask_intensity] = [0, 0, 255]  # Синим выделяем газон
    plt.imshow(image_with_lawn_intensity)
    plt.title(f'Выделение газона (интенсивность)\nРегионов: {len(regions_intensity)}')
else:
    plt.imshow(image_rgb)
    plt.title('Не удалось выделить регионы (интенсивность)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Анализ результатов
print(f"Выделено регионов (цветной критерий): {len(regions_color)}")
print(f"Выделено регионов (критерий интенсивности): {len(regions_intensity)}")

if regions_color:
    total_area_color = sum(np.sum(region) for region in regions_color)
    print(f"Общая площадь (цветной критерий): {total_area_color} пикселей")

if regions_intensity:
    total_area_intensity = sum(np.sum(region) for region in regions_intensity)
    print(f"Общая площадь (интенсивность): {total_area_intensity} пикселей")

# Часть 2: Сравнение различных критериев однородности
print("\n=== Часть 2: Сравнение критериев однородности ===")

# Создаем тестовое изображение для сравнения критериев
test_image = np.random.rand(100, 100) * 100
test_image[30:70, 30:70] = 150 + np.random.rand(40, 40) * 20

test_seeds = [(50, 50)]

# Применяем разные критерии
rg_test = RegionGrowing()

print("Тестируем критерий интенсивности...")
result_intensity = rg_test.grow_regions(test_image, test_seeds, 
                                      rg_test.homogeneity_criterion_intensity, 
                                      threshold=25, use_color=False)

print("Тестируем критерий дисперсии...")
result_variance = rg_test.grow_regions(test_image, test_seeds,
                                     rg_test.homogeneity_criterion_variance,
                                     threshold=30, use_color=False)

print("Тестируем критерий градиента...")
result_gradient = rg_test.grow_regions(test_image, test_seeds,
                                     rg_test.homogeneity_criterion_gradient,
                                     threshold=15, use_color=False)

# Визуализация сравнения критериев
plt.figure(figsize=(18, 5))

plt.subplot(1, 5, 1)
plt.imshow(test_image, cmap='gray')
plt.title('Тестовое изображение')
plt.axis('off')

plt.subplot(1, 5, 2)
if result_intensity:
    plt.imshow(result_intensity[0], cmap='hot')
    plt.title('Критерий интенсивности')
else:
    plt.imshow(np.zeros_like(test_image), cmap='hot')
    plt.title('Нет результата (интенсивность)')
plt.axis('off')

plt.subplot(1, 5, 3)
if result_variance:
    plt.imshow(result_variance[0], cmap='hot')
    plt.title('Критерий дисперсии')
else:
    plt.imshow(np.zeros_like(test_image), cmap='hot')
    plt.title('Нет результата (дисперсия)')
plt.axis('off')

plt.subplot(1, 5, 4)
if result_gradient:
    plt.imshow(result_gradient[0], cmap='hot')
    plt.title('Критерий градиента')
else:
    plt.imshow(np.zeros_like(test_image), cmap='hot')
    plt.title('Нет результата (градиент)')
plt.axis('off')

# Сравнение площади регионов
area_intensity = np.sum(result_intensity[0]) if result_intensity else 0
area_variance = np.sum(result_variance[0]) if result_variance else 0
area_gradient = np.sum(result_gradient[0]) if result_gradient else 0

plt.subplot(1, 5, 5)
methods = ['Интенсивность', 'Дисперсия', 'Градиент']
areas = [area_intensity, area_variance, area_gradient]
colors = ['blue', 'orange', 'green']
plt.bar(methods, areas, color=colors)
plt.title('Сравнение площади регионов')
plt.ylabel('Площадь (пикселей)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print(f"Площадь региона (критерий интенсивности): {area_intensity}")
print(f"Площадь региона (критерий дисперсии): {area_variance}")
print(f"Площадь региона (критерий градиента): {area_gradient}")

# Часть 3: Watershed + Distance Transform для подсчета пальмовых деревьев
print("\n=== Часть 3: Подсчет пальмовых деревьев (Watershed) ===")

# Создаем изображение с пальмами для демонстрации
palm_image = np.ones((300, 400), dtype=np.uint8) * 100

# Добавляем пальмы (темные объекты на светлом фоне)
palm_positions = [(80, 100), (150, 120), (220, 90), (300, 130), (350, 100)]
for i, (x, y) in enumerate(palm_positions):
    radius = 15 + np.random.randint(-3, 3)
    cv2.circle(palm_image, (x, y), radius, 50, -1)
    # Добавляем небольшие вариации
    cv2.circle(palm_image, (x-5, y-5), 5, 40, -1)
    cv2.circle(palm_image, (x+5, y+5), 5, 40, -1)

# Применяем watershed алгоритм
def watershed_segmentation(image):
    # Бинаризация
    thresh = filters.threshold_otsu(image)
    binary = image < thresh
    
    # Удаляем мелкие шумы
    cleaned = morphology.remove_small_objects(binary, min_size=20)
    
    # Distance transform
    distance = ndimage.distance_transform_edt(cleaned)
    
    # Находим маркеры
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), 
                              labels=cleaned)
    markers = measure.label(local_maxi)
    
    # Применяем watershed
    labels = watershed(-distance, markers, mask=cleaned)
    
    return labels, distance, cleaned

# Применяем алгоритм
print("Применяем Watershed сегментацию...")
labels, distance, binary_mask = watershed_segmentation(palm_image)

# Подсчитываем объекты
props = measure.regionprops(labels)
# Игнорируем фон (метка 0)
palm_count = len([prop for prop in props if prop.area > 10])

# Визуализация результатов
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(palm_image, cmap='gray')
plt.title('Исходное изображение пальм')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Бинаризованное изображение')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(distance, cmap='hot')
plt.title('Distance Transform')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(labels, cmap='nipy_spectral')
plt.title(f'Watershed сегментация\nНайдено объектов: {palm_count}')
plt.axis('off')

# Визуализация с контурами
plt.subplot(2, 3, 5)
plt.imshow(palm_image, cmap='gray')
valid_regions = [prop for prop in props if prop.area > 10]
for region in valid_regions:
    y, x = region.centroid
    plt.plot(x, y, 'r+', markersize=15, markeredgewidth=2)
    # Рисуем bounding box
    minr, minc, maxr, maxc = region.bbox
    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                         fill=False, edgecolor='red', linewidth=2)
    plt.gca().add_patch(rect)
plt.title('Обнаруженные пальмы')
plt.axis('off')

# Сравнение с ожидаемым количеством
plt.subplot(2, 3, 6)
expected = len(palm_positions)
detected = palm_count
accuracy = detected / expected * 100 if expected > 0 else 0

categories = ['Ожидалось', 'Обнаружено']
values = [expected, detected]
colors = ['blue', 'green']

plt.bar(categories, values, color=colors, alpha=0.7)
plt.ylabel('Количество')
plt.title(f'Точность: {accuracy:.1f}%')
for i, v in enumerate(values):
    plt.text(i, v + 0.1, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

print(f"Ожидаемое количество пальм: {len(palm_positions)}")
print(f"Обнаружено пальм: {palm_count}")
print(f"Точность обнаружения: {accuracy:.1f}%")

# Дополнительная информация
print("\n=== Рекомендации по настройке параметров ===")
print("1. Для алгоритма разрастания регионов:")
print("   - Цветной критерий: threshold=20-40")
print("   - Критерий интенсивности: threshold=15-30") 
print("   - Критерий дисперсии: threshold=20-35")
print("   - Используйте 4-8 семян равномерно распределенных по газону")
print("\n2. Для Watershed сегментации:")
print("   - Настройте min_size для remove_small_objects()")
print("   - Экспериментируйте с размерами footprint для peak_local_max")
print("   - Применяйте морфологические операции для улучшения бинаризации")
