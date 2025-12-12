import cv2
import sys
import os

def mosaic_area(image, x, y, w, h, ratio=0.05):
    """주어진 영역을 픽셀화(모자이크) 처리"""
    face = image[y:y+h, x:x+w]
    if face.size == 0:
        return image
    small = cv2.resize(face, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = mosaic
    return image

def mosaic_image(input_path, output_path="images/output_mosaic.jpg", ratio=0.05):
    if not os.path.exists(input_path):
        print(f"Error: 입력 이미지가 없습니다: {input_path}")
        return False

    img = cv2.imread(input_path)
    if img is None:
        print("Error: 이미지 로드 실패 (파일형식 확인)")
        return False

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        print("Error: haarcascade XML 파일을 찾을 수 없습니다.")
        print("-> OpenCV의 haarcascades 경로에서 파일을 찾지 못했습니다.")
        return False

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error: Cascade 로드 실패")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)

    if len(faces) == 0:
        print("경고: 이미지에서 얼굴을 찾지 못했습니다.")
    else:
        for (x, y, w, h) in faces:
            mosaic_area(img, x, y, w, h, ratio=ratio)

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    success = cv2.imwrite(output_path, img)
    if success:
        print(f"모자이크 적용 완료: {output_path}")
    else:
        print("Error: 이미지 저장 실패")
    return success

if __name__ == "__main__":
    default_input = os.path.join("images", "input.jpg")
    default_output = os.path.join("images", "output_mosaic.jpg")

    if len(sys.argv) >= 2:
        inp = sys.argv[1]
    else:
        inp = default_input

    if len(sys.argv) >= 3:
        outp = sys.argv[2]
    else:
        outp = default_output

    mosaic_image(inp, outp)
