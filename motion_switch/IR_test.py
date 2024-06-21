import cv2

# IR 카메라가 일반적으로 1번 디바이스로 할당됨 (0은 기본 웹캠)
cap = cv2.VideoCapture(1)

# 카메dl라가 제대로 열렸는지 확인
if not cap.isOpened():
    print("Camera not accessible")
    exit()

while True:
    # 프레임을 계속 읽기
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # IR 이미지는 흑백으로 출력되므로 컬러 변환 필요 없음
    cv2.imshow('IR Camera Output', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
