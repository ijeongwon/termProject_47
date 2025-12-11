import cv2
import numpy as np

def main():
    # 1. 이미지 읽기
    img = cv2.imread("images/input.jpg")
    if img is None:
        print("이미지를 찾지 못했습니다. 경로와 파일명을 확인하세요.")
        return

    # 2. 그레이스케일 + 블러
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 에지 검출 (Canny)
    edges = cv2.Canny(blur, 50, 150)

    # 4. 외곽선(contour) 찾기 - 가장 바깥쪽만
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        print("외곽선을 찾지 못했습니다.")
        return

    # 5. 가장 큰(contourArea가 제일 큰) 외곽선 하나 선택
    main_contour = max(contours, key=cv2.contourArea)

    # 6. 원본 이미지 복사 후, 초록색으로 외곽선 그리기
    result = img.copy()
    cv2.drawContours(result, [main_contour], -1, (0, 255, 0), 2)

    # 7. 외곽선 점들을 기준으로 직선 피팅해서 중심 방향 선 그리기
    vx, vy, x0, y0 = cv2.fitLine(
        main_contour, cv2.DIST_L2, 0, 0.01, 0.01
    )

    rows, cols = result.shape[:2]

    # fitLine으로 얻은 직선을 이미지 왼쪽 끝~오른쪽 끝까지 연장
    left_y = int((-x0 * vy / vx) + y0)
    right_y = int(((cols - x0) * vy / vx) + y0)

    cv2.line(
        result,
        (0, left_y),
        (cols - 1, right_y),
        (0, 0, 255),   # 빨간색
        2
    )

    # 8. 결과 저장
    cv2.imwrite("images/output.jpg", result)
    print("완료! images/output.jpg 파일이 생성되었습니다.")

if __name__ == "__main__":
    main()
