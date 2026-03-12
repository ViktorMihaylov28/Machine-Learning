from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import prepare_dataset_from_folders, load_image


def train_cat_dog_classifier(cat_folder, dog_folder):
    # Подготвя данните за обучение от папките с котки и кучета, като ги преоразмерява до 64x64 пиксела и ги нормализира.
    X, y = prepare_dataset_from_folders(cat_folder, dog_folder, size=(64, 64))

    if len(X) == 0:
        print("Няма намерени снимки за обучение.")
        return None

    if len(X) < 6:
        print("Данните са твърде малко. Добави още снимки.")
        return None

    # Проверява дали има и от двата класа - Снимки на котки (0) и кучета (1) 
    cat_count = sum(1 for value in y if value == 0)
    dog_count = sum(1 for value in y if value == 1)

    if cat_count == 0 or dog_count == 0:
        print("Трябва да има поне една котка и поне едно куче.")
        return None

    # Разделя данните на обучаващи и тестови набори, като запазва пропорциите на класовете (стратифицирано разделяне), за да се гарантира, че и в двата набора има представителство на котки и кучета.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # KNN е лесен за обяснение и е подходящ за ученически проект - той класифицира нови данни въз основа на сходството с обучаващите данни.
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nРезултати от класификацията:")
    print(f"Точност: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["котка", "куче"]))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


def predict_single_image(model, image_path):
    # Разпознава една нова снимка на котка или куче, като я зарежда, преоразмерява до 64x64 пиксела, нормализира и използва обучената KNN модел за предсказване на класа (котка или куче).
    image = load_image(image_path, size=(64, 64))

    if image is None:
        return

    features = image.flatten().reshape(1, -1) / 255.0
    prediction = model.predict(features)[0]

    if prediction == 0:
        print("Прогнозата е: котка")
    else:
        print("Прогнозата е: куче")