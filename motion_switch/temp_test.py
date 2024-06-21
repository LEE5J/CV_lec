import cv2
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = deeplabv3_resnet50(pretrained=True).to(device)
model.eval()
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cap = cv2.VideoCapture(0)


def get_segmentation_mask(frame):
    input_tensor = preprocess(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    return output.argmax(0).byte().cpu().numpy()


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = get_segmentation_mask(frame_rgb)
        person_mask = mask == 15  # Person class ID
        contours, _ = cv2.findContours(person_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segmented_frame = np.zeros_like(frame)  # 초기에 모든 픽셀을 검은색으로 설정

        for contour in contours:
            if cv2.contourArea(contour) > 100:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Check orientation
                width, height = rect[1]
                if width < height:
                    # Standing: White mask
                    cv2.drawContours(segmented_frame, [box], 0, (255, 255, 255), -1)  # 채워진 흰색 사각형 그리기
                else:
                    # Lying: Black mask
                    cv2.drawContours(segmented_frame, [box], 0, (0, 0, 0), -1)  # 채워진 검은색 사각형 그리기

                # Draw bounding box
                cv2.drawContours(segmented_frame, [box], 0, (0, 255, 0), 2)
                angle = rect[2]
                cv2.putText(segmented_frame, f"Angle: {angle:.2f} degrees", (int(rect[0][0]), int(rect[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Segmented Frame', segmented_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
