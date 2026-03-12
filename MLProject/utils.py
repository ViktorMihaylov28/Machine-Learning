import os
import cv2
import numpy as np


def fix_path(path_text):
    # Маха излишни интервали и кавички от пътя, ако има такива (може да се случи при копиране от терминала или от други източници)
    path_text = path_text.strip().strip('"').strip("'")

    # Оправя наклонените черти в зависимост от операционната система, за да се избегнат проблеми при зареждането на файлове (особено при Windows, където се използват обратни наклонени черти)
    path_text = path_text.replace("\\", os.sep)
    path_text = path_text.replace("/", os.sep)

    return path_text


def find_existing_image_path(image_path):
    # Оправя пътя и проверява дали файлът съществува, ако не - пробва с други разширения (ако е подадено без разширение или с грешно разширение)
    image_path = fix_path(image_path)

    # Ако файлът съществува, връща го директно (това е най-честият случай, когато потребителят е подал правилен път) - това е оптимално, защото не прави излишни проверки
    if os.path.exists(image_path):
        return image_path

    # Ако не съществува, пробва и други разширения (това е полезно, ако потребителят е подал път без разширение или с грешно разширение, например "data/test/1" вместо "data/test/1.jpeg") - това е по-бавно, но помага да се намери файлът, ако има малка грешка в разширението
    base, ext = os.path.splitext(image_path)
    possible_extensions = [".jpeg", ".jpg", ".png", ".bmp"]

    if ext:
        ordered_extensions = [ext.lower()] + [e for e in possible_extensions if e != ext.lower()]
    else:
        ordered_extensions = possible_extensions

    for current_ext in ordered_extensions:
        new_path = base + current_ext
        if os.path.exists(new_path):
            return new_path

    return None


def read_image_unicode_safe(image_path):
    # Тази функция чете снимка по по-сигурен начин, като използва numpy и OpenCV, което помага да се избегнат проблеми с кодировката на пътищата,
    # особено ако в пътя има кирилица или други специални символи, които могат да причинят грешки при стандартното зареждане на изображения с OpenCV.
    try:
        file_bytes = np.fromfile(image_path, dtype=np.uint8)
        if file_bytes.size == 0:
            return None

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def load_image(image_path, size=None):
    # Намира истинския път до снимката (оправя пътя и проверява дали файлът съществува, ако не - пробва с други разширения) и я зарежда, като използва метод, който работи по-добре при кирилица. Ако е зададен размер, прави resize на снимката. Тази функция е основна за зареждането на снимки в проекта и се използва от други функции, които работят със снимки, като например тези за класификация и клъстеризация. Тя гарантира, че снимките ще бъдат заредени правилно, дори ако има проблеми с пътищата или кодировката.
    real_path = find_existing_image_path(image_path)

    if real_path is None:
        print("Грешка: файлът не беше намерен.")
        print("Подаден път:", image_path)
        return None

    # Чете снимката с метод, който работи по-добре при кирилица (особено ако пътят до снимката съдържа кирилица или други специални символи, които могат да причинят проблеми при стандартното зареждане на изображения с OpenCV) - това помага да се избегнат грешки при зареждането на снимки от папки с кирилски имена или от пътища, които съдържат специални символи.
    image = read_image_unicode_safe(real_path)

    if image is None:
        print("Грешка: снимката не можа да се отвори.")
        print("Намерен файл:", real_path)
        return None

    # OpenCV чете в BGR, затова го прави в RGB (това е важно, защото повечето други библиотеки и инструменти за обработка на изображения използват RGB формат, така че това гарантира съвместимост и правилно визуализиране на цветовете в снимките) - това е стандартна практика при работа с OpenCV, за да се гарантира, че цветовете в снимките са правилно интерпретирани и визуализирани.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ако е зададен размер, прави resize на снимката (това е полезно, за да се стандартизират размерите на снимките, особено когато се използват за обучение на модели, които изискват вход с фиксиран размер) - това помага да се гарантира, че всички снимки в набора от данни имат еднакъв размер, което е важно за ефективното обучение на модели за машинно обучение и клъстеризация.
    if size is not None:
        image = cv2.resize(image, size)

    return image


def get_image_paths_from_folder(folder):
    # Взима всички снимки от папката (търси само файлове с разширения .jpeg, .jpg, .png, .bmp) и връща списък с пътищата до тях. Тази функция е полезна за събиране на данни от папки, които съдържат снимки, и се използва от други функции, които работят със снимки, като например тези за класификация и клъстеризация. Тя гарантира, че ще бъдат взети само валидни снимки и че те ще бъдат подредени в ясен ред.
    folder = fix_path(folder)
    image_paths = []

    if not os.path.exists(folder):
        return image_paths

    valid_extensions = (".jpeg", ".jpg", ".png", ".bmp")

    for file_name in os.listdir(folder):
        if file_name.lower().endswith(valid_extensions):
            full_path = os.path.join(folder, file_name)
            image_paths.append(full_path)

    # Подрежда ги, за да са в ясен ред (това помага да се гарантира, че снимките ще бъдат обработвани в предвидим ред, което може да бъде полезно за отстраняване на грешки и за по-лесно проследяване на данните) - това е особено полезно, когато се използват папки с много снимки, тъй като подреденият списък прави по-лесно намирането и идентифицирането на конкретни снимки в процеса на обучение или клъстеризация.
    image_paths.sort()

    return image_paths


def extract_simple_features(image):
    # Вади лесни характеристики за клъстеризация (средна стойност на R, G, B каналите, средна яркост, процент светли пиксели и процент тъмни пиксели, както и размерите на снимката) - тези характеристики са избрани, защото са лесни за изчисление и могат да предоставят полезна информация за различията между снимките, което може да помогне при клъстеризацията. Те са базови характеристики, които могат да се използват като вход за алгоритми за машинно обучение или клъстеризация, за да се групират снимките въз основа на техните визуални свойства.
    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mean_brightness = np.mean(gray)
    light_pixels = np.sum(gray > 170) / gray.size
    dark_pixels = np.sum(gray < 85) / gray.size

    height, width, _ = image.shape

    return [
        mean_r,
        mean_g,
        mean_b,
        mean_brightness,
        light_pixels,
        dark_pixels,
        width,
        height
    ]


def prepare_dataset_from_folders(cat_folder, dog_folder, size=(64, 64)):
    # Подготвя данните за обучение котка/куче от папките с котки и кучета, като ги преоразмерява до 64x64 пиксела и ги нормализира. Тази функция е основна за подготовката на данните за обучение на модела за класификация на котки и кучета. Тя гарантира, че снимките ще бъдат заредени правилно, ще бъдат преоразмерени до един и същ размер и ще бъдат нормализирани, което е важно за ефективното обучение на модела. Тя също така проверява дали има достатъчно данни от двата класа (котки и кучета) и пропуска снимки, които не могат да бъдат заредени, като предоставя информация за това.
    X = []
    y = []

    cat_folder = fix_path(cat_folder)
    dog_folder = fix_path(dog_folder)

    cat_paths = get_image_paths_from_folder(cat_folder)
    dog_paths = get_image_paths_from_folder(dog_folder)

    print(f"Намерени котки: {len(cat_paths)}")
    print(f"Намерени кучета: {len(dog_paths)}")

    # Котка = 0
    for path in cat_paths:
        image = load_image(path, size=size)
        if image is not None:
            features = image.flatten() / 255.0
            X.append(features)
            y.append(0)
        else:
            print("Пропусната снимка:", path)

    # Куче = 1
    for path in dog_paths:
        image = load_image(path, size=size)
        if image is not None:
            features = image.flatten() / 255.0
            X.append(features)
            y.append(1)
        else:
            print("Пропусната снимка:", path)

    return np.array(X), np.array(y)


def prepare_unlabeled_dataset(folder_list, size=(64, 64)):
    # Подготвя данните за обучение без учител от посочените папки, като ги преоразмерява до 64x64 пиксела и ги нормализира. Тази функция е основна за подготовката на данните за клъстеризация на снимки без учител. Тя гарантира, че снимките ще бъдат заредени правилно, ще бъдат преоразмерени до един и същ размер и ще бъдат нормализирани, което е важно за ефективната клъстеризация на снимките. Тя също така пропуска снимки, които не могат да бъдат заредени, като предоставя информация за това. Тя връща масив с характеристиките на снимките и списък с имената на снимките, което може да бъде полезно за идентифициране на снимките в резултатите от клъстеризацията.
    X = []
    names = []

    for folder in folder_list:
        folder = fix_path(folder)
        paths = get_image_paths_from_folder(folder)

        for path in paths:
            image = load_image(path, size=size)
            if image is not None:
                features = extract_simple_features(image)
                X.append(features)
                names.append(os.path.basename(path))
            else:
                print("Пропусната снимка:", path)

    return np.array(X), names