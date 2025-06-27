import cv2
import numpy as np
from gaze_model.gaze_model_data import GazeModels


class DrawGaze:
    """
    웹캠 실시간 시선 추정 및 시각화
    OpenCV로 웹캠을 열고, 얼굴/눈/머리포즈/시선 추정 결과를 실시간으로 시각화합니다.
    """
    def draw_gaze_vector(self, frame, eye_center, gaze_vector, scale=60, color=(0, 0, 255)):
        """
            시선 벡터를 프레임에 그려줍니다.
        """
        x, y = int(eye_center[0]), int(eye_center[1])
        dx, dy = int(gaze_vector[0]*scale), int(-gaze_vector[1]*scale)
        cv2.arrowedLine(frame, (x, y), (x+dx, y+dy), color, 2, tipLength=0.3)

    def crop_eye(self, center, w, h, frame, eye_size=60):
        """
            눈 크롭
        """
        x, y = center
        x1, y1 = max(0, x-eye_size//2), max(0, y-eye_size//2)
        x2, y2 = min(w, x+eye_size//2), min(h, y+eye_size//2)
        eye = frame[y1:y2, x1:x2]

        if eye.shape[0] != eye_size or eye.shape[1] != eye_size:
            eye = cv2.resize(eye, (eye_size, eye_size))

        return eye.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def draw_gaze_by_webcam(self, gaze_models: GazeModels):
        """
            웹캠 정보를 활용하여 시야 검출.
        """
        cap = cv2.VideoCapture(0)
        w, h = 640, 480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 얼굴 검출
            input_blob = cv2.resize(frame, (672, 384))
            input_blob = input_blob.transpose(2, 0, 1)[np.newaxis, ...]
            input_blob = input_blob.astype(np.float32)
            face_results = gaze_models.face([input_blob])[gaze_models.face.outputs[0]]
            faces = face_results[0][0]
            for det in faces:
                conf = det[2]
                if conf < 0.5:
                    continue
                xmin, ymin, xmax, ymax = (det[3:7] * np.array([w, h, w, h])).astype(int)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # 2. 랜드마크 추정 (눈 위치)
                face_img = frame[ymin:ymax, xmin:xmax]
                if face_img.size == 0:
                    continue
                lm_input = cv2.resize(face_img, (48, 48)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
                lm_results = gaze_models.landmarks([lm_input])[gaze_models.landmarks.outputs[0]]
                lm = lm_results.reshape(-1, 2)
                left_eye = (int(lm[0][0]*(xmax-xmin))+xmin, int(lm[0][1]*(ymax-ymin))+ymin)
                right_eye = (int(lm[1][0]*(xmax-xmin))+xmin, int(lm[1][1]*(ymax-ymin))+ymin)
                cv2.circle(frame, left_eye, 3, (255, 0, 0), -1)
                cv2.circle(frame, right_eye, 3, (0, 0, 255), -1)

                # 3. 머리포즈 추정
                hp_input = cv2.resize(face_img, (60, 60)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
                hp_results = gaze_models.head_pose([hp_input])
                head_pose = [hp_results[o][0][0] for o in gaze_models.head_pose.outputs]

                # 4. 시선 추정
                left_eye_img = self.crop_eye(left_eye, w, h, frame)
                right_eye_img = self.crop_eye(right_eye, w, h, frame)
                head_pose_np = np.array(head_pose, dtype=np.float32).reshape(1, 3)
                gaze_result = gaze_models.gaze([left_eye_img, right_eye_img, head_pose_np])
                gaze_vector = gaze_result[gaze_models.gaze.outputs[0]][0]

                # 5. 시선 벡터 시각화
                eye_center = ((left_eye[0]+right_eye[0])//2, (left_eye[1]+right_eye[1])//2)
                self.draw_gaze_vector(frame, eye_center, gaze_vector)

            cv2.imshow("Gaze Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
