# 유사도 계산하것
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
from sklearn.metrics.pairwise import cosine_similarity

from captcha.GLCM import extract_glcm
from captcha.LBP import extract_lbp
from captcha.cnn import CNN
from captcha.ensemble_ev import extract_color_histogram, extract_harris_corner, extract_sift, prepare_feature_data
from captcha.utility import load_target_image, load_data_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def cnn_features(images,model):
    model.eval()
    features=[]
    # 이미지 리스트를 텐서로 변환
    images_tensor = torch.stack([torch.tensor(img, dtype=torch.float32) for img in images])
    images_tensor = images_tensor.permute(0, 3, 1, 2)
    images_tensor = images_tensor.to(device)  # GPU로 이동 (device는 'cuda' 또는 'cpu')

    # TensorDataset 생성
    test_dataset = TensorDataset(images_tensor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 배치 크기를 1로 설정


    for (image,) in test_loader:
        image = image.to(device)  # 배치 차원 추가 및 디바이스 할당
        with torch.no_grad():  # 그라디언트 계산 비활성화
            output = model.feature(image)
        features.append(output.cpu().numpy().flatten())  # CPU로 이동 후 numpy 배열로 변환
    return features

def find_best_similarity(t_features, features,t_filenames,challange="c1",atmp=0):
    top_10_indices = []
    for t_feature in t_features:
        cosine_similarities = cosine_similarity([t_feature], features)[0]
        top_10_indices.append(np.argsort(-cosine_similarities)[:10])
    top_10_file_names = [[None] * 10 for _ in range(len(top_10_indices))]
    for i, row in enumerate(top_10_indices):
        for j, index in enumerate(row):
            top_10_file_names[i][j] = file_names[index]
    array1 = np.array(top_10_file_names)
    array2 = np.array(t_filenames).reshape((-1, 1))
    df = pd.DataFrame(np.hstack((array2, array1)), columns=['Target Filename'] + [f'Top {i + 1}' for i in range(10)])
    df.to_csv(f'{challange}_t2_a{atmp}.csv', index=False)
    return top_10_indices

if __name__ == '__main__':
    images, file_names = load_data_image("Data")
    t_images,t_filenames = load_target_image('target')
    # t_features = np.array([extract_glcm(image) for image in t_images])
    # features = np.array([extract_glcm(image) for image in images])
    # top_10_indices = find_best_similarity(t_features,features,t_filenames)
    feature_extractors = [
        ('Color Histogram', extract_color_histogram),  # {'C': 100, 'gamma': 0.1} 53%
        ('LBP', extract_lbp),  # {'C': 100, 'gamma': 10} 38.5
        ('GLCM', extract_glcm),  # 50프로 까지 끌어올림 {'C': 1, 'gamma': 0.001} 36.8
        ('Harris Corner', extract_harris_corner),  # {'C': 1, 'gamma': 0.001} 33
        ('SIFT', extract_sift),
        # ('HOG', extract_hog) # 시간 너무 오래걸림 성능으 좋음
    ]
    _t_features = prepare_feature_data(t_images,feature_extractors)
    _features = prepare_feature_data(images,feature_extractors)
    for i in range(3):
        model = lgb.Booster(model_file=f'asemble{i}.model')
        t_features = model.predict(_t_features)
        features = model.predict(_features)
        top_10_indices = find_best_similarity(t_features, features, t_filenames,challange="c1",atmp=i)

    # for _i in range(3):
    #     model = CNN().to(device)
    #     state_dict = torch.load(f"cnn{_i}.pth")
    #     model.load_state_dict(state_dict)
    #     features = cnn_features(images,model)
    #     t_features = cnn_features(t_images,model)
    #     top_10_indices = find_best_similarity(t_features, features,t_filenames,challange="c2",atmp=_i)







