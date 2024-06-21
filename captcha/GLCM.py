import time
import pickle
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from captcha.utility import load_images_from_folder, load_target_image, print_t1_result
import lightgbm as lgb

def extract_glcm(image, distances=[5], angles=[0]):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        return np.hstack([contrast, energy, homogeneity])
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def parallel_extract_glcm(images, num_workers=8):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # functools.partial을 사용하여 extract_glcm 함수에 고정 매개변수 설정
        partial_extract_glcm = partial(extract_glcm, distances=[5], angles=[0])
        # 이미지 목록에 대해 함수 병렬 실행
        results = list(executor.map(partial_extract_glcm, images))
    return results

def train_GLCM(X_train,y_train,name ='GLCM_model.txt' ):
   train_data = lgb.Dataset(X_train, label=y_train)
   lgbparams = {
       'device_type': 'gpu',
       'boosting_type': 'gbdt',
       'objective': 'multiclass',
       'num_class': 10, # 고유 라벨 수에 맞게 클래스 수를 설정
       'num_leaves': 31,
       'learning_rate': 0.05,
       'metric': 'multi_logloss'
   }
   model = lgb.train(lgbparams, train_data, num_boost_round=300)
   model.save_model(name)
   return model

def test_GLCM(feature,model=None,name ='GLCM_model.txt'):
    if model is None:
        model = lgb.Booster(model_file=name)
    return model.predict(feature)


def analysis(model):
    importance = model.feature_importance(importance_type='split')
    feature = model.feature_name()
    feature_importance = dict(zip(feature, importance))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features:
        print(f"Feature: {feature}, Importance: {importance}")


if __name__ == '__main__':
    start = time.time()
    images, labels = load_images_from_folder('DATA')
    print("LOADING img", time.time() - start)
    features = np.array([extract_glcm(image) for image in images])
    # features = parallel_extract_glcm(images, num_workers=8)
    print("upto glcm extract", time.time() - start)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.01)
    for i in range(3):
        model = train_GLCM(X_train, y_train,f"GLCM_model{i}.txt")
        # y_pred = test_GLCM(X_test,model)# y_pred는 각 클래스에 대한 확률을 반환하므로, 가장 높은 확률을 가진 클래스를 선택
        # y_pred = np.argmax(y_pred, axis=1)
        # 예측된 라벨을 사용하여 정확도 계산
        # accuracy = accuracy_score(y_test, y_pred)
        # print(f'lgb Accuracy: {accuracy:.3f}')
        # print("Total time:", time.time() - start)
        # analysis(model)
        images, filenames = load_target_image('target')
        features = np.array(parallel_extract_glcm(images, num_workers=8))
        preds = test_GLCM(features,model)
        preds = np.argmax(preds, axis=1)
        print_t1_result(filenames, preds, attmp=i)
