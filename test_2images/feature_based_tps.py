import cv2
import numpy as np
import matplotlib.pyplot as plt

def register_with_tps_warp(fixed_np, moving_np):
    """
    特徴点マッチングとThin Plate Spline(TPS)で非剛体位置合わせ
    """
    print("特徴点ベースの非剛体ワープを開始")

    # STEP 1: 特徴点マッチング 
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(fixed_np, None)
    kp2, des2 = akaze.detectAndCompute(moving_np, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("特徴点が少なすぎてTPSが作れない…")
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des2, des1, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10: # 最低でも10点はほしい
        print("イイ感じのペアが少なすぎる…")
        return None, None
    
    # マッチングした点の座標を取得
    src_pts = np.array([kp2[m.queryIdx].pt for m in good_matches])
    dst_pts = np.array([kp1[m.trainIdx].pt for m in good_matches])

    # STEP 2: Thin Plate Spline (TPS) で変形ルールを作る！
    # ここが新しいパート！
    print("  TPSの変形ルールを作成中…！✨")
    
    # createThinPlateSplineShapeTransformerに座標を渡す
    tps = cv2.createThinPlateSplineShapeTransformer()
    
    # マッチングした点をsshape(source shape)とtshape(target shape)として渡す
    # OpenCVの仕様で(1, N, 2)の形にする必要がある
    sshape = src_pts.reshape(1, -1, 2)
    tshape = dst_pts.reshape(1, -1, 2)
    
    matches_for_tps = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    tps.estimateTransformation(tshape, sshape, matches_for_tps)

    # STEP 3: 作った変形ルールで画像をワープ
    h, w = fixed_np.shape
    print("  画像をワープ中…！")
    registered_np = tps.warpImage(moving_np)

    # マッチング結果の描画用
    match_img = cv2.drawMatches(moving_np, kp2, fixed_np, kp1, good_matches, None, flags=cv2.DrawMatchesFlags.NOT_DRAW_SINGLE_POINTS)

    return registered_np, match_img

if __name__ == '__main__':
    fixed_image_path = "/Users/horiieikkei/Desktop/VS code/test_2images/test2slices/cropped_MMP_109_x4_largest copy.tif"
    moving_image_path = "/Users/horiieikkei/Desktop/VS code/test_2images/test2slices/cropped_MMP_110_x4_largest copy.tif"

    # 画像を読み込む
    fixed_np = cv2.imread(fixed_image_path, cv2.IMREAD_GRAYSCALE)
    moving_np = cv2.imread(moving_image_path, cv2.IMREAD_GRAYSCALE)

    registered_image, match_image = register_with_tps_warp(fixed_np, moving_np)
        
    if registered_image is not None:
        # 結果表示
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("最終兵器TPS！", fontsize=20)
        axes[0, 0].imshow(moving_np, cmap='gray')
        axes[0, 0].set_title("Original Moving Image", fontsize=14)
        axes[0, 1].imshow(fixed_np, cmap='gray')
        axes[0, 1].set_title("Fixed Image", fontsize=14)
        axes[1, 0].imshow(match_image)
        axes[1, 0].set_title("Feature Matches (このペアを信用)", fontsize=14)
        axes[1, 1].imshow(registered_image, cmap='gray')
        axes[1, 1].set_title("Final Result (TPS Warp)", fontsize=14)
        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()