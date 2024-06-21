import os
import time
from concurrent.futures import ProcessPoolExecutor
import lightgbm as lgb
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils import shuffle
from tqdm import tqdm

from captcha.GLCM import extract_glcm, test_GLCM, parallel_extract_glcm
from captcha.LBP import extract_lbp, parallel_extract_lbp
from captcha.utility import load_images_from_folder, label_mapping, num2label, grid_search, grid_search_cuda, train_lgb, \
    load_target_image, print_t1_result
from sklearn.preprocessing import LabelEncoder

# 1. 특징 추출 함수 구현
def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()



def extract_harris_corner(image, max_corners=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        corners = corners.reshape(-1, 2)
        # Flatten the corners and pad with zeros if there are fewer than max_corners
        flattened_corners = corners.flatten()
        if len(flattened_corners) < max_corners * 2:
            padding = np.zeros(max_corners * 2 - len(flattened_corners))
            flattened_corners = np.hstack((flattened_corners, padding))
        return flattened_corners
    else:
        return np.zeros(max_corners * 2)


def extract_sift(image, max_features=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    if des is not None:
        # Sort keypoints and descriptors by the response (strength) of the keypoints
        sorted_indices = np.argsort([kp[i].response for i in range(len(kp))])[::-1]
        sorted_des = des[sorted_indices]
        # Select the top max_features descriptors
        if len(sorted_des) < max_features:
            padding = np.zeros((max_features - len(sorted_des), des.shape[1]))
            sorted_des = np.vstack((sorted_des, padding))
        return sorted_des[:max_features].flatten()
    else:
        return np.zeros(max_features * 128)


def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define HOGDescriptor parameters
    winSize = (100, 100)
    blockSize = (20, 20)
    blockStride = (10, 10)
    cellSize = (10, 10)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog_features = hog.compute(gray)
    return hog_features.flatten()


def extract_feature_for_images(images, extractor,name):
    start_time = time.time()
    if name != "LBP":
        features = [extractor(image) for image in images]
    else:
        with ProcessPoolExecutor(max_workers=8) as executor:
            features = list(executor.map(extractor, images))
    print(f"{name}: {time.time() - start_time:.4f} sec")
    return np.vstack(features)


def prepare_feature_data(images, feature_extractors, num_workers=4):
    feature_list = []
    for name, extractor in feature_extractors:
        combined_features = extract_feature_for_images(images, extractor,name)
        feature_list.append(combined_features)
    # 모든 특징을 수평으로 결합
    combined_features = np.hstack(feature_list)
    return combined_features


# 3. 모델 훈련 및 예측
def train_and_predict(train_data, train_labels, test_data, test_labels):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data)
    X_test_scaled = scaler.transform(test_data)

    # 커널 PCA 수행
    kpca = KernelPCA(n_components=20, kernel='rbf', gamma=0.01)
    X_train_kpca = kpca.fit_transform(X_train_scaled)
    X_test_kpca = kpca.transform(X_test_scaled)

    # SVM 모델 생성 및 학습
    model = SVC(kernel='rbf', C=1.0, gamma='auto')
    model.fit(X_train_kpca, train_labels)

    svm = SVC(kernel='rbf')
    svm.fit(train_data, train_labels)
    origin_preds = svm.predict(test_data)
    print("origin accuracy:", accuracy_score(test_labels, origin_preds))

    # model = SVC(kernel='rbf', gamma=0.5, C=1.0)
    # model.fit(train_data, train_labels)
    preds = model.predict(X_test_kpca)
    accuracy = accuracy_score(test_labels, preds)
    print("accuracy /w kerner trick:", accuracy)

    return model ,accuracy, preds


# 4. 앙상블 및 결과 평가
def ensemble_predictions(prediction_sets, prob_sets):
    final_preds = []
    for i in tqdm(range(len(prediction_sets[0]))):
        combined_probs = np.mean([probs[i] for probs in prob_sets], axis=0)
        final_pred = np.argmax(combined_probs)
        final_preds.append(final_pred)
    return final_preds


# 5. 결과 저장 및 정확도 출력
def save_results(true_labels, preds, output_file='results.csv'):
    preds_encoded = [num2label[label] for label in preds]
    results = pd.DataFrame({
        'TrueLabel': true_labels,
        'PredictedLabel': preds,
        'Correct': np.array(true_labels) == np.array(preds)
    })
    results.to_csv(output_file, index=False)
    accuracy = accuracy_score(true_labels, preds)
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy

# 메인 실행 함수
def main(folder):
    # images, labels = load_images_from_folder(folder)
#Best parameters: {'C': 10, 'gamma': 0.001} /w rbf
    feature_extractors = [
        ('Color Histogram', extract_color_histogram),#{'C': 100, 'gamma': 0.1} 53%
        ('LBP', extract_lbp),#{'C': 100, 'gamma': 10} 38.5
        ('GLCM', extract_glcm),# 50프로 까지 끌어올림 {'C': 1, 'gamma': 0.001} 36.8
        ('Harris Corner', extract_harris_corner),#{'C': 1, 'gamma': 0.001} 33
        ('SIFT', extract_sift),
        # ('HOG', extract_hog) # 시간 너무 오래걸림 성능으 좋음
    ]
    # features = prepare_feature_data(images, feature_extractors)
    # model = train_lgb(train_data, train_labels,"Color_his_model.txt")
    # grid_search_cuda(train_data, train_labels)
    # Train and predict using SVM
    # model ,accuracy, probs = train_and_predict(train_data, train_labels, test_data,test_labels)
    # preds = model.predict(test_data)
    # accuracy = accuracy_score(test_labels, preds)
    # print(f'Accuracy: {accuracy:.4f}')
    # Save results
    # save_results(test_labels, probs)
    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.02)
    for i in range(3):
    # model = train_lgb(X_train, y_train, f"asemble{i}.model")
        model = lgb.Booster(model_file=f'asemble{i}.model')
        # preds = model.predict(X_test)
        #     # preds = np.argmax(preds, axis=1)
        #     # acc = accuracy_score(y_test, preds)
        #     # print("acc:" + str(acc))
        images, filenames = load_target_image('target')
        features = np.array(prepare_feature_data(images,feature_extractors))
        preds = model.predict(features)
        preds = np.argmax(preds, axis=1)
        print_t1_result(filenames, preds, attmp=i)


# 폴더 경로를 main 함수에 전달하여 실행
if __name__ == "__main__":
    main('data')

