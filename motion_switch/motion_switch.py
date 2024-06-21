import cv2
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 모델 로드 및 평가 모드 설정
model = deeplabv3_resnet50(pretrained=True).to(device)
model.eval()
# 이미지 전처리를 위한 변환 함수
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# 웹캠 캡처 객체 생성
cap = cv2.VideoCapture(0)
def get_segmentation_mask(frame):
    input_tensor = preprocess(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    return output.argmax(0).byte().cpu().numpy()  # 사람에 해당하는 마스크 반환
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = get_segmentation_mask(frame_rgb)
        person_mask = mask == 15  # 사람 클래스 ID
        # 마스크에서 경계 찾기
        contours, _ = cv2.findContours(person_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmented_frame = np.zeros_like(frame)
        segmented_frame[person_mask] = frame[person_mask]
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # 작은 객체는 무시
                rect = cv2.minAreaRect(contour)  # 최소 면적 사각형 계산
                box = cv2.boxPoints(rect)  # 사각형의 4개 꼭짓점
                box = np.int0(box)
                cv2.drawContours(segmented_frame, [box], 0, (0, 255, 0), 2)  # 사각형 그리기
                # 방향 표시
                angle = rect[2]
                cv2.putText(segmented_frame, f"Angle: {angle:.2f} degrees", (int(rect[0][0]), int(rect[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 결과 표시
        cv2.imshow('Segmented Frame', segmented_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()