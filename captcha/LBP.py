import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from captcha.utility import load_images_from_folder, load_target_image, print_t1_result


def extract_lbp(image, num_points=24, radius=8):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # 이미지가 흑백일 경우, 직접 gray 변수에 할당
        gray = image
    lbp = local_binary_pattern(gray, num_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist


def parallel_extract_lbp(images, num_workers=8):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(extract_lbp, images))
    return results

def train_lbp(train_data, labels,C=10, gamma=0.01):
    scaler = StandardScaler()
    X_train_lbp_scaled = scaler.fit_transform(train_data)
    svm = SVC(kernel='rbf', C=C, gamma=gamma)
    svm.fit(X_train_lbp_scaled, labels)
    return svm

def test_lbp(images, model=None):
    if model is None:
        pass
    features = parallel_extract_lbp(images)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    preds = model.predict(features_scaled)
    return preds


if __name__ == '__main__':
    start = time.time()
    images, labels = load_images_from_folder('DATA')
    print("LOADING img", time.time() - start)
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
    features = np.array(parallel_extract_lbp(X_train, num_workers=8))
    print("extracting img", time.time() - start)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # param_grid = {
    #     'C': [0.1, 1, 10, 100],
    #     'gamma': [0.001, 0.01, 0.1, 1, 10]
    # }
    # svc = SVC(kernel='rbf')
    # # 그리드 서치 실행
    # grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(features_scaled, y_train) #Best parameters: {'C': 10, 'gamma': 0.01}
    # # 결과 출력
    # print("Best parameters:", grid_search.best_params_)
    # print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    # model = train_lbp(features,y_train)
    # preds = test_lbp(X_test, model)
    # accuracy = accuracy_score(X_test, preds)
    # print(accuracy)
    images, filenames = load_target_image('target')
    features = np.array(parallel_extract_lbp(images, num_workers=8))
    features_scaled = scaler.fit_transform(features)
    preds = test_lbp(features_scaled)
    print_t1_result(features, preds, attmp=0)


