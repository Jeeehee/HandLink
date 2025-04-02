import cv2
import mediapipe as mp
import numpy as np
import time, os

# last update: 25.04.02 - Jee

# Guide Text
guide = "[ Guide ]"
guide_inputWord = "Enter the sign language words you want to collect.\n(separated by commas.)"
guide_setDefaultWords = "No word was entered. Set to default words."  # ❤️
guide_preparing = "Preparing to collect. Collect 3 sets of 10 seconds each."
guide_start = "Start collecting sign language data." 
guide_remainingTime = "seconds left until collection ends"
guide_collectionCompleted = "Sign language collection completed"
guide_savedCompleted = "Data saved completed"

class SignLanguageDatasetBuilder:

    def __init__(self):
        self.number_of_hands = 2 # 인식할 손 개수
        self.seq_length = 40 # 한 시퀀스의 길이
        self.secs_for_action = 10 # 각 동작을 수집할 시간(초)

        # MediaPipe의 Hands 모듈 초기화
        self.mp_hands = mp.solutions.hands  # 손 검출 모델 불러오기
        self.mp_drawing = mp.solutions.drawing_utils # 손 관절을 화면에 그리는 유틸리티
        self.hands = self.mp_hands.Hands(
        max_num_hands = self.number_of_hands, # 최대 인식 가능한 손 개수 설정
        min_detection_confidence=0.5, # 손 인식 최소 신뢰도 설정
        min_tracking_confidence=0.5  # 손 추적 최소 신뢰도 설정
        )

        # 데이터 저장 설정
        self.created_time = int(time.time())  # 데이터 저장 시 사용될 시간 값 생성
        os.makedirs('dataset', exist_ok=True)  # 데이터 저장할 폴더 생성 (이미 존재하면 무시)
        
        # 카메라 설정
        self.cap = None

    # 카메라 설정
    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open camera")
            self.cap = None
            return
        
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    # 사용자로부터 액션 입력받기
    def request_action_input(self):
        print(guide, guide_inputWord)
        input_actions = input().split(',')
        return [action.strip() for action in input_actions if action.strip()]

    # 이미지 처리 및 손 인식
    def detect_hands_from_image(self, img):
        img = cv2.flip(img, 1)  # 좌우반전 (거울 효과)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV → MediaPipe 용 (BGR → RGB)
        result = self.hands.process(img_rgb) # MediaPipe에서 손 인식 수행
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) # MediaPipe → OpenCV 용 (RGB → BGR)
        return img, result
    
     # 손이 감지된 경우 특성 추출
    def process_hand_landmarks(self, result, idx, img):
        data = [] # 수집된 데이터를 저장할 빈 리스트 생성
        
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:  # 감지된 손에 대해 반복 수행
                # 관절 좌표 추출
                joint = np.zeros((21, 4))  # 21개 관절에 대해 (x, y, z, visibility) 초기화
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                
                # 관절 간 벡터 계산
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
                v = v2 - v1  # [20, 3] 벡터 차이 계산
                
                # 벡터 정규화
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                
                # 관절 각도 계산
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
                angle = np.degrees(angle)
                
                # 데이터 레이블링 및 저장
                angle_label = np.array([angle], dtype=np.float32)  # 각도 데이터를 NumPy 배열로 변환
                angle_label = np.append(angle_label, idx)  # 동작(action) 인덱스 추가
                
                # 전체 특성 벡터 생성
                d = np.concatenate([joint.flatten(), angle_label])
                data.append(d)
                
                # 손 관절 시각화
                self.mp_drawing.draw_landmarks(img, res, self.mp_hands.HAND_CONNECTIONS)
        return data, img
    
    def save_data(self, data, action, attempt):
        if not data:
            print(f"No data collected for {action}, attempt {attempt + 1}")
            return
         
        # 수집된 데이터 저장 - 원본 데이터
        data = np.array(data)  # 수집된 데이터를 NumPy 배열로 변환
        print(f"[ {action} {attempt + 1} ] - 데이터 크기: {data.shape}")
        np.save(os.path.join('dataset', f'raw_{action}_{attempt + 1}_{self.created_time}'), data)
        
        # 시퀀스 데이터 생성 및 저장
        full_seq_data = []  # 시퀀스 데이터를 저장할 리스트 생성
        for seq in range(len(data) - self.seq_length):  # 시퀀스 길이만큼 슬라이딩 윈도우 생성
            full_seq_data.append(data[seq:seq + self.seq_length])
        
        full_seq_data = np.array(full_seq_data)  # NumPy 배열로 변환
        print(f"[ {action} {attempt + 1} ] - 시퀀스 데이터 크기: {full_seq_data.shape}")
        np.save(os.path.join('dataset', f'seq_{action}_{attempt + 1}_{self.created_time}'), full_seq_data)
        print(f"[ {action} {attempt + 1} ] - 데이터 저장 성공")

    def collect_sign_language_data(self, action, idx, attempt):
        collected_data = []
        
        # 준비 화면 표시
        ret, img = self.cap.read()  # 카메라에서 프레임 읽기

        if not ret or img is None:
            print("카메라에서 이미지를 읽지 못했습니다.")
            return collected_data
    
        img = cv2.flip(img, 1)  # 좌우반전 (거울 효과)

        # 데이터 수집 전 사용자에게 대기 메시지 표시
        cv2.putText(
            img, 
            f'{guide} {guide_preparing}', 
            org=(20, 50), 
            fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=1, 
            color=0, 
            thickness=2
        )

        cv2.imshow('Sign Language Data Collection', img)  # 현재 프레임 화면에 표시
        cv2.waitKey(3000)  # 3초(3 000ms) 대기 - 사용자가 준비할 시간

        # 데이터 수집 시작
        print(f"Collecting data for '{action}' -Attempt {attempt + 1} Start")
        
        start_time = time.time()
        fail_count = 0
        max_failures = 10  # 실패 허용 횟수
    
        # 데이터 수집 루프 - 설정된 시간 동안 데이터 수집
        while time.time() - start_time < self.secs_for_action:
            ret, img = self.cap.read() # 카메라에서 프레임 읽기
            if not ret:
                fail_count += 1
                if fail_count >= max_failures:
                    print("Error: Too many frame capture failures.")
                    break
                continue
                
            img, result = self.detect_hands_from_image(img)
            frame_data, img = self.process_hand_landmarks(result, idx, img)
            collected_data.extend(frame_data)
            
            # 남은 시간 계산 및 표시
            elapsed_time = time.time() - start_time
            remaining_time = int(max(0, self.secs_for_action - elapsed_time))
            
            if remaining_time > 0:
                # 수집 진행 중 메시지
                cv2.putText(
                    img, 
                    f'{guide} Attempt {attempt + 1} - {remaining_time} {guide_remainingTime}', 
                    org=(20, 50), 
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=1, 
                    color=0, 
                    thickness=2
                )
            else:
                # 수집 완료 메시지
                cv2.putText(
                    img, 
                    f'{guide} {guide_collectionCompleted}', 
                    org=(20, 50), 
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=1, 
                    color=0, 
                    thickness=2
                )

            # 화면 표시 및 종료 확인
            cv2.imshow('Sign Language Data Collection', img)

            if cv2.waitKey(1) == ord('q'):
                break
                
        return collected_data
    
    def run(self):
        # 사용자로부터 액션 입력받기
        actions = self.request_action_input()

         # 카메라 설정
        self.setup_camera()

        if self.cap is not None and self.cap.isOpened():
            for idx, action in enumerate(actions):
                for attempt in range(3):  # 한 동작당 세 번 반복하여 학습
                    # 데이터 수집
                    data = self.collect_sign_language_data(action, idx, attempt)

                    # 데이터 저장
                    self.save_data(data, action, attempt)
                    
                    print(f"Attempt {attempt + 1} Done")
        
            # 자원 해제
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
        else:
            print("Camera could not be opened")

        # 수어 리스트 저장
        data_as_strings = [",".join(actions)]
        with open("signLanguageList/list.csv", "a", encoding="utf-8-sig") as f:
            np.savetxt(f, data_as_strings, fmt="%s")

# 메인 실행 코드
if __name__ == "__main__":
    recognizer = SignLanguageDatasetBuilder()
    recognizer.run()