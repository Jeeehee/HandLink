import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['안녕', '반짝이다', '가다']
number_of_hands = 2 # 인식할 손 개수
seq_length = 40 # 한 시퀀스의 길이
secs_for_action = 10 # 각 동작을 수집할 시간(초)

# MediaPipe의 Hands 모듈 초기화
mp_hands = mp.solutions.hands  # 손 검출 모델 불러오기
mp_drawing = mp.solutions.drawing_utils # 손 관절을 화면에 그리는 유틸리티
hands = mp_hands.Hands(
    max_num_hands=number_of_hands, # 최대 인식 가능한 손 개수 설정
    min_detection_confidence=0.5, # 손 인식 최소 신뢰도 설정
    min_tracking_confidence=0.5) # 손 추적 최소 신뢰도 설정

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # ❤️ FPS를 30으로 설정
created_time = int(time.time())  # 데이터 저장 시 사용될 시간 값 생성
os.makedirs('dataset', exist_ok=True) # 데이터 저장할 폴더 생성 (이미 존재하면 무시)

while cap.isOpened(): # 웹캠이 정상적으로 열려 있을 때 실행
    for idx, action in enumerate(actions): # 각 제스처(action)에 대해 반복, idx는 인덱스 번호
        for attempt in range(3):  # ❤️ 한 동작당 세 번 반복하여 학습
            print(f"Attempt {attempt + 1} Start")
            data = []  # 수집된 데이터를 저장할 빈 리스트 생성

            ret, img = cap.read() # 카메라에서 프레임 읽기 (ret: 성공 여부, img: 프레임)

            img = cv2.flip(img, 1)  # 좌우반전 (거울 효과)

            # 데이터 수집 전 사용자에게 대기 메시지 표시
            cv2.putText(img, f'Waiting for collecting', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            cv2.imshow('img', img) # 현재 프레임 화면에 표시
            cv2.waitKey(5000) # 5초(5000ms) 대기 - 사용자가 준비할 시간

            start_time = time.time() # 데이터 수집 시작 시간 기록

            while time.time() - start_time < secs_for_action: # 설정된 시간 동안 데이터 수집
                ret, img = cap.read() # 카메라에서 프레임 읽기

                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV → MediaPipe 용 (BGR → RGB)
                result = hands.process(img) # MediaPipe에서 손 인식 수행
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # MediaPipe → OpenCV 용 (RGB → BGR)

                if result.multi_hand_landmarks is not None: # 손이 감지된 경우 실행
                    for res in result.multi_hand_landmarks: # 감지된 손에 대해 반복 수행
                        joint = np.zeros((21, 4)) # 21개 관절에 대해 (x, y, z, visibility) 초기화
                        for j, lm in enumerate(res.landmark): # 각 관절에 대한 좌표 추출
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
                        v = v2 - v1 # [20, 3]
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                        angle = np.degrees(angle)

                        angle_label = np.array([angle], dtype=np.float32) # 각도 데이터를 NumPy 배열로 변환
                        angle_label = np.append(angle_label, idx) # 동작(action) 인덱스 추가

                        d = np.concatenate([joint.flatten(), angle_label])

                        data.append(d)

                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'): # 'q' 키를 누르면 데이터 수집 중지
                    break

            data = np.array(data) # 수집된 데이터를 NumPy 배열로 변환
            print(action, attempt + 1, data.shape) # 현재 동작 및 데이터 크기 출력
            np.save(os.path.join('dataset', f'raw_{action}_{attempt + 1}_{created_time}'), data)  # 원본 데이터 저장

            full_seq_data = [] # 시퀀스 데이터를 저장할 리스트 생성
            for seq in range(len(data) - seq_length): # 시퀀스 길이만큼 슬라이딩 윈도우 생성
                full_seq_data.append(data[seq:seq + seq_length])

            full_seq_data = np.array(full_seq_data) # NumPy 배열로 변환
            print(action, attempt + 1, full_seq_data.shape) # 시퀀스 데이터 크기 출력
            np.save(os.path.join('dataset', f'seq_{action}_{attempt + 1}_{created_time}'), full_seq_data)

            print(f"Attempt {attempt + 1} Done")
            
    break  # 첫 번째 반복 후 종료