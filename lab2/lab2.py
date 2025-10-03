import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

# Создаем тестовое изображение (или загружаем существующее)
try:
    image = cv2.imread('sar_1.jpg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError
except:
    print("Файл не найден, создаем тестовое изображение")
    # Создаем тестовое изображение с текстом для лучшей визуализации
    image = np.ones((256, 256), dtype=np.uint8) * 128
    cv2.putText(image, 'TEST', (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    cv2.putText(image, 'IMAGE', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)

print(f"Размер изображения: {image.shape}")

# Функции для добавления шумов
def add_gaussian_noise(image, mean=0, sigma=25):
    """Добавляет гауссов шум"""
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """Добавляет солевой и перечный шум"""
    noisy = image.copy()
    # Солевой шум (белые точки)
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy[salt_mask] = 255
    # Перечный шум (черные точки)
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy[pepper_mask] = 0
    return noisy

def add_uniform_noise(image, intensity=50):
    """Добавляет равномерный шум"""
    noise = np.random.uniform(-intensity, intensity, image.shape)
    noisy = image.astype(float) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Функции фильтрации
def apply_median_filter(image, kernel_size=3):
    """Медианный фильтр"""
    return cv2.medianBlur(image, kernel_size)

def apply_gaussian_filter(image, kernel_size=5, sigma=1.0):
    """Гауссов фильтр"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Билатеральный фильтр"""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_nlm_filter(image, h=10, template_window_size=7, search_window_size=21):
    """Фильтр нелокальных средних"""
    return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)

# Функции для оценки качества
def calculate_mse(img1, img2):
    """Mean Squared Error"""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

def calculate_psnr(img1, img2):
    """Peak Signal-to-Noise Ratio"""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Structural Similarity Index"""
    # Упрощенная реализация SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    return np.mean(numerator / denominator)

# 1. Добавление шумов
print("\n1. ДОБАВЛЕНИЕ ШУМОВ")
print("=" * 50)

# Гауссов шум
gaussian_noisy = add_gaussian_noise(image, sigma=30)
print(f"Гауссов шум: MSE = {calculate_mse(image, gaussian_noisy):.2f}")

# Солевой и перечный шум
salt_pepper_noisy = add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02)
print(f"Солевой/перечный шум: MSE = {calculate_mse(image, salt_pepper_noisy):.2f}")

# Равномерный шум
uniform_noisy = add_uniform_noise(image, intensity=40)
print(f"Равномерный шум: MSE = {calculate_mse(image, uniform_noisy):.2f}")

# 2. Тестирование фильтров с различными параметрами
print("\n2. ТЕСТИРОВАНИЕ ФИЛЬТРОВ")
print("=" * 50)

def test_filters(noisy_image, original_image, noise_type):
    """Тестирует различные фильтры на зашумленном изображении"""
    
    results = {}
    
    # Медианный фильтр с разными размерами ядра
    print(f"\n{noise_type} - Медианный фильтр:")
    for ksize in [3, 5, 7]:
        filtered = apply_median_filter(noisy_image, ksize)
        mse = calculate_mse(original_image, filtered)
        psnr = calculate_psnr(original_image, filtered)
        ssim = calculate_ssim(original_image, filtered)
        results[f'median_{ksize}'] = {'mse': mse, 'psnr': psnr, 'ssim': ssim, 'image': filtered}
        print(f"  Размер ядра {ksize}: MSE={mse:.2f}, PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    
    # Гауссов фильтр с разными параметрами
    print(f"\n{noise_type} - Гауссов фильтр:")
    for ksize, sigma in [(3, 0.5), (5, 1.0), (7, 1.5)]:
        filtered = apply_gaussian_filter(noisy_image, ksize, sigma)
        mse = calculate_mse(original_image, filtered)
        psnr = calculate_psnr(original_image, filtered)
        ssim = calculate_ssim(original_image, filtered)
        results[f'gaussian_{ksize}_{sigma}'] = {'mse': mse, 'psnr': psnr, 'ssim': ssim, 'image': filtered}
        print(f"  Ядро {ksize}x{ksize}, σ={sigma}: MSE={mse:.2f}, PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    
    # Билатеральный фильтр с разными параметрами
    print(f"\n{noise_type} - Билатеральный фильтр:")
    for d, sigma_c, sigma_s in [(5, 50, 50), (9, 75, 75), (15, 100, 100)]:
        filtered = apply_bilateral_filter(noisy_image, d, sigma_c, sigma_s)
        mse = calculate_mse(original_image, filtered)
        psnr = calculate_psnr(original_image, filtered)
        ssim = calculate_ssim(original_image, filtered)
        results[f'bilateral_{d}_{sigma_c}_{sigma_s}'] = {'mse': mse, 'psnr': psnr, 'ssim': ssim, 'image': filtered}
        print(f"  d={d}, σ_color={sigma_c}, σ_space={sigma_s}: MSE={mse:.2f}, PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    
    # Фильтр нелокальных средних с разными параметрами
    print(f"\n{noise_type} - Фильтр нелокальных средних:")
    for h in [5, 10, 20]:
        filtered = apply_nlm_filter(noisy_image, h=h)
        mse = calculate_mse(original_image, filtered)
        psnr = calculate_psnr(original_image, filtered)
        ssim = calculate_ssim(original_image, filtered)
        results[f'nlm_h{h}'] = {'mse': mse, 'psnr': psnr, 'ssim': ssim, 'image': filtered}
        print(f"  h={h}: MSE={mse:.2f}, PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    
    return results

# Тестируем на разных типах шумов
gaussian_results = test_filters(gaussian_noisy, image, "Гауссов шум")
salt_pepper_results = test_filters(salt_pepper_noisy, image, "Солевой/перечный шум")
uniform_results = test_filters(uniform_noisy, image, "Равномерный шум")

# 3. Визуализация результатов
print("\n3. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 50)

def plot_comparison(original, noisy, filtered_results, noise_type, best_filters):
    """Визуализирует сравнение фильтров"""
    
    plt.figure(figsize=(20, 12))
    
    # Исходное и зашумленное изображение
    plt.subplot(2, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title(f'{noise_type}\nMSE: {calculate_mse(original, noisy):.2f}')
    plt.axis('off')
    
    # Лучшие фильтры по каждому типу
    for i, (filter_name, filter_data) in enumerate(best_filters[:3]):
        plt.subplot(2, 4, i + 3)
        plt.imshow(filter_data['image'], cmap='gray')
        plt.title(f'{filter_name}\nMSE: {filter_data["mse"]:.2f}, SSIM: {filter_data["ssim"]:.4f}')
        plt.axis('off')
    
    # Гистограммы
    plt.subplot(2, 4, 7)
    plt.hist(original.ravel(), bins=50, alpha=0.7, label='Исходное', color='blue')
    plt.hist(noisy.ravel(), bins=50, alpha=0.7, label='Зашумленное', color='red')
    plt.legend()
    plt.title('Гистограммы')
    
    plt.subplot(2, 4, 8)
    for filter_name, filter_data in best_filters[:3]:
        plt.hist(filter_data['image'].ravel(), bins=50, alpha=0.7, label=filter_name)
    plt.legend()
    plt.title('Гистограммы фильтров')
    
    plt.tight_layout()
    plt.show()

# Находим лучшие фильтры для каждого типа шума
def find_best_filters(results, metric='ssim'):
    """Находит лучшие фильтры по заданной метрике"""
    sorted_filters = sorted(results.items(), key=lambda x: x[1][metric], reverse=(metric != 'mse'))
    return sorted_filters

print("\nЛУЧШИЕ ФИЛЬТРЫ:")
print("-" * 30)

# Для гауссова шума
best_gaussian = find_best_filters(gaussian_results, 'ssim')
print(f"\nГауссов шум (лучшие по SSIM):")
for i, (name, data) in enumerate(best_gaussian[:3]):
    print(f"  {i+1}. {name}: SSIM={data['ssim']:.4f}, MSE={data['mse']:.2f}")

# Для солевого/перечного шума
best_salt_pepper = find_best_filters(salt_pepper_results, 'ssim')
print(f"\nСолевой/перечный шум (лучшие по SSIM):")
for i, (name, data) in enumerate(best_salt_pepper[:3]):
    print(f"  {i+1}. {name}: SSIM={data['ssim']:.4f}, MSE={data['mse']:.2f}")

# Для равномерного шума
best_uniform = find_best_filters(uniform_results, 'ssim')
print(f"\nРавномерный шум (лучшие по SSIM):")
for i, (name, data) in enumerate(best_uniform[:3]):
    print(f"  {i+1}. {name}: SSIM={data['ssim']:.4f}, MSE={data['mse']:.2f}")

# Визуализируем результаты
plot_comparison(image, gaussian_noisy, gaussian_results, "Гауссов шум", best_gaussian)
plot_comparison(image, salt_pepper_noisy, salt_pepper_results, "Солевой/перечный шум", best_salt_pepper)
plot_comparison(image, uniform_noisy, uniform_results, "Равномерный шум", best_uniform)

# 4. Сводная таблица результатов
print("\n4. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 50)

def print_summary_table(results_dict, noise_type):
    """Печатает сводную таблицу результатов"""
    print(f"\n{noise_type}:")
    print("Фильтр\t\t\tMSE\tPSNR\tSSIM")
    print("-" * 50)
    
    for filter_name, metrics in results_dict.items():
        if 'median' in filter_name or 'gaussian' in filter_name or 'bilateral' in filter_name or 'nlm' in filter_name:
            print(f"{filter_name:20} {metrics['mse']:6.2f} {metrics['psnr']:6.2f} {metrics['ssim']:6.4f}")

print_summary_table(gaussian_results, "ГАУССОВ ШУМ")
print_summary_table(salt_pepper_results, "СОЛЕВОЙ/ПЕРЕЧНЫЙ ШУМ")
print_summary_table(uniform_results, "РАВНОМЕРНЫЙ ШУМ")

# 5. Анализ и выводы
print("\n5. ВЫВОДЫ И РЕКОМЕНДАЦИИ")
print("=" * 50)

print("\nАНАЛИЗ ЭФФЕКТИВНОСТИ ФИЛЬТРОВ:")
print("-" * 40)

# Анализ для каждого типа шума
noise_types = {
    "Гауссов шум": (gaussian_results, best_gaussian),
    "Солевой/перечный шум": (salt_pepper_results, best_salt_pepper),
    "Равномерный шум": (uniform_results, best_uniform)
}

for noise_name, (results, best) in noise_types.items():
    print(f"\n{noise_name}:")
    best_filter = best[0]
    print(f"  Лучший фильтр: {best_filter[0]}")
    print(f"  Показатели: MSE={best_filter[1]['mse']:.2f}, "
          f"PSNR={best_filter[1]['psnr']:.2f} dB, "
          f"SSIM={best_filter[1]['ssim']:.4f}")
    
    # Анализ эффективности по типам фильтров
    filter_types = {}
    for name, data in results.items():
        filter_type = name.split('_')[0]
        if filter_type not in filter_types:
            filter_types[filter_type] = []
        filter_types[filter_type].append(data['ssim'])
    
    print("  Средний SSIM по типам фильтров:")
    for ftype, ssims in filter_types.items():
        avg_ssim = np.mean(ssims)
        print(f"    {ftype}: {avg_ssim:.4f}")

print("\nОБЩИЕ РЕКОМЕНДАЦИИ:")
print("1. Для гауссова шума: фильтр нелокальных средних или билатеральный фильтр")
print("2. Для солевого/перечного шума: медианный фильтр")
print("3. Для равномерного шума: фильтр нелокальных средних")
print("4. Медианный фильтр эффективен против импульсных шумов")
print("5. Билатеральный фильтр сохраняет границы лучше других")
print("6. Фильтр нелокальных средних дает наилучшее качество, но требует больше вычислений")

print("\nЗадание выполнено!")
