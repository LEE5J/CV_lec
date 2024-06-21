import os
import time
# from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import pandas as pd
import contextlib
import joblib
import torch
from PIL import Image
# from PIL.Image import Image
# from cuml import train_test_split
# from cuml.common.device_selection import using_device_type
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from torchvision import transforms
from torchvision.datasets import folder
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.utils import shuffle


transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomVerticalFlip(p=0.5),  # 50% 확률로 수직 뒤집기
    transforms.RandomRotation(degrees=15),  # -15도에서 15도 사이로 무작위 회전
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_mapping = {
    'Bicycle': 0,
    'Bridge': 1,
    'Bus': 2,
    'Car': 3,
    'Chimney': 4,
    'Crosswalk': 5,
    'Hydrant': 6,
    'Motorcycle': 7,
    'Palm': 8,
    'Traffic Light': 9
}

num2label = {v: k for k, v in label_mapping.items()}


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def preprocess_image(image_path, size, use_gaussian=True, use_median=True):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is not None:
        if use_gaussian:
            image = cv2.GaussianBlur(image, (5, 5), 0)
        if use_median:
            image = cv2.medianBlur(image, 5)
        resized_image = cv2.resize(image, size)
        return resized_image
    return None


# CNN 전용 이미지 로더
def process_and_load_images(images):
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(image) for image in images]
    stk = [transform_train(image) for image in images]
    tensor_images = torch.stack(stk)  # 이미지 리스트를 텐서 스택으로 변환
    return tensor_images


# CNN 실전용 이미지 로더
def process_and_load_targets(images):
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(image) for image in images]
    stk = [transform_test(image) for image in images]
    tensor_images = torch.stack(stk)  # 이미지 리스트를 텐서 스택으로 변환
    return tensor_images


def load_images_from_folder(folder, size=(128, 128), max_img=-1):
    classes = os.listdir(folder)
    images = []
    labels = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_image = {}
        for class_label in classes:
            class_dir = os.path.join(folder, class_label)
            image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if
                           f.endswith(('.png', '.jpg', '.jpeg'))][:max_img]
            for image_file in image_files:
                future = executor.submit(preprocess_image, image_file, size)
                future_to_image[future] = label_mapping[class_label]
        for future in as_completed(future_to_image):
            image = future.result()
            if image is not None:
                images.append(image)
                labels.append(future_to_image[future])
    images, labels = shuffle(images, labels, random_state=42)
    return images, labels


def load_target_image(folder, size=(128, 128), max_img=-1):
    images = []
    filenames = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_image = {}
        image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))][
                      :max_img]
        for image_file in image_files:
            future = executor.submit(preprocess_image, image_file, size)
            future_to_image[future] = image_file
        for future in as_completed(future_to_image):
            image = future.result()
            if image is not None:
                images.append(image)
                filenames.append(future_to_image[future])  # 이미지 파일명을 리스트에 추가
    return images, filenames  # 이미지와 이미지 파일명을 함께 반환


def load_data_image(folder, size=(128, 128), max_img=-1):
    classes = os.listdir(folder)
    images = []
    filenames = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_image = {}
        for class_label in classes:
            class_dir = os.path.join(folder, class_label)
            image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if
                           f.endswith(('.png', '.jpg', '.jpeg'))][:max_img]
            for image_file in image_files:
                future = executor.submit(preprocess_image, image_file, size)
                future_to_image[future] = image_file
        for future in as_completed(future_to_image):
            image = future.result()
            if image is not None:
                images.append(image)
                filenames.append(future_to_image[future])
    return images, filenames

def print_t1_result(filename, preds, challenge="c1", attmp=1):
    pred_names = [num2label[int(pred)] for pred in preds]
    results_df = pd.DataFrame({
        'filename': filename,
        'prediction': pred_names
    })
    results_df.to_csv(f'{challenge}_t1_a{attmp}.csv', index=False)



# SVM 최적값 탐색용
def grid_search(features, labels):
    param_grid = {
        # 'kernel': ['rbf', 'poly', 'sigmoid'],
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10]
    }
    svc = SVC()
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=6)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    grid_search.fit(features_scaled, labels)  #Best parameters: {'C': 10, 'gamma': 0.01}
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


def grid_search_cuda(features, labels):
    features_np = np.array(features, dtype=float)
    labels_np = np.array(labels, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(features_np, labels_np, test_size=0.2)
    param_grid = {
        # 'kernel': ['rbf', 'poly', 'sigmoid'],
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10]
    }
    best_score = 0
    best_params = {}
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            with using_device_type('gpu'):
                model = SVC(C=C, gamma=gamma, kernel='rbf')
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            print(f"Test with C={C}, gamma={gamma}, Accuracy={accuracy:.4f}")
            if accuracy > best_score:
                best_score = accuracy
                best_params = {'C': C, 'gamma': gamma}
    print("Best parameters:", best_params)
    print("Best accuracy: {:.4f}".format(best_score))


def submit_files_c1_t1(filename):
    images , filenames = load_target_image(folder)
    features = parallel_extract_glcm(images, num_workers=8)
    y_pred = test_GLCM(features)
    y_pred = np.argmax(y_pred, axis=1)
    print_t1_result(filename, y_pred, challenge="c1", attmp=1)



def submit_files_c1_t2(labels, filename):
    pass


def submit_files_c2_t1(labels, filename):
    pass


def submit_files_c2_t2(labels, filename):
    pass

def train_lgb(X_train, y_train,name):
    train_data = lgb.Dataset(X_train, label=y_train)
    lgbparams = {
        'device_type': 'gpu',
        'boosting_type': 'dart',
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),  # 고유 라벨 수에 맞게 클래스 수를 설정
        'metric': 'multi_logloss',  # 성능 측정 메트릭
    }
    model = lgb.train(lgbparams, train_data, num_boost_round=100)
    model.save_model(name)
    return model

def test_lgb(X_test, y_test,filename):
    model = lgb.Booster(model_file=filename)
    model.predict(X_test)
    return model



#
# if __name__ == '__main__':
#     train_data = lgb.Dataset(X_train, label=y_train)
#     test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
