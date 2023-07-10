import cv2
import numpy as np

# マウスイベントのコールバック関数
def mouse_callback(event, x, y, flags, param):
    global points, selected_point

    if event == cv2.EVENT_LBUTTONDOWN:
        # マウス左ボタンが押されたら最も近い頂点を選択
        selected_point = get_nearest_point(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        # マウス左ボタンが離されたら頂点の位置を更新
        points[selected_point] = (x, y)

def get_nearest_point(x, y):
    # 最も近い頂点のインデックスを返す
    distances = [np.linalg.norm(np.array(point) - np.array((x, y))) for point in points]
    print(distances)
    return np.argmin(distances)


# カメラの読込み
# 内蔵カメラがある場合、下記引数の数字を変更する必要あり
cap = cv2.VideoCapture(0)
#1フレーム毎　読込み
ret, image = cap.read()
# カメラを閉じる

# 画像のサイズを取得
height, width = image.shape[:2]

# 画像の4頂点の初期位置を設定
points = [(0, 0), (0, height - 1), (width - 1, height - 1),(width - 1, 0),]
selected_point = None

# ウィンドウを作成し、マウスイベントのコールバック関数を設定
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# ホモグラフィー変換を行うための4点を定義
src_points = np.float32(points)
dst_points = np.float32([[0, 0],  [0, height - 1],[width - 1, height - 1], [width - 1, 0]])


while True:
    # 画像をコピーして描画
    img_copy = image.copy()

    # 頂点に丸を描く
    for point in points:
        cv2.circle(img_copy, point, 5, (0, 0, 255), -1)

    # 選択された頂点があれば、その頂点をドラッグ中のマウスの位置に移動
    if selected_point is not None:
        cv2.circle(img_copy, points[selected_point], 5, (0, 255, 0), -1)

    # 画像を表示
    cv2.imshow("Image", img_copy)

    # ESCキーが押されたら終了
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

# ホモグラフィー変換を行うための4点を定義
src_points = np.float32(points)
dst_points = np.float32([[0, 0],  [0, height - 1],[width - 1, height - 1], [width - 1, 0]])
# ホモグラフィー行列を計算
homography_matrix= cv2.getPerspectiveTransform(dst_points, src_points)


# 動画終了まで、1フレームずつ読み込んで表示する。
while(cap.isOpened()):
    # 1フレーム毎　読込み
    ret, frame = cap.read()

    # ホモグラフィー変換を適用
    warped_image = cv2.warpPerspective(frame, homography_matrix, (width, height))
    # GUIに表示
    cv2.imshow("Camera", warped_image)
    # qキーが押されたら途中終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()



