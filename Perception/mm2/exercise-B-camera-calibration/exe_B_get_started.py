import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import os

# Prepare object points (0,0,0), (1,0,0), ... scaled to square_size
board_size = (10, 7)               # internal corners: cols, rows
square_size = 20.0                 # mm (adjust to your square size)

pts_obj = np.zeros((board_size[0]*board_size[1], 3), np.float32)
pts_obj[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
pts_obj *= square_size

# Save object points template (optional)
np.savez('obj-pts.npz', points=pts_obj)

# Containers for calibration
objpoints = []   # 3d points in real world space
imgpoints = []   # 2d points in image plane

# Termination criteria for sub-pixel refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Read images
image_files = sorted(glob.glob(os.path.join('images', '*.jpeg')))
if len(image_files) == 0:
    raise SystemExit("No images found in 'images' folder. Place .jpeg files there and retry.")

# Iterate and detect corners
for fname in image_files:
    img = cv.imread(fname)
    if img is None:
        print(f"Warning: couldn't read {fname}, skipping.")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # findChessboardCorners expects patternSize = (cols, rows)
    found, corners = cv.findChessboardCorners(
        gray, board_size,
        flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        # refine corner locations
        corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(pts_obj)            # same object points for every view
        imgpoints.append(corners_refined)

        # optional: draw and save visualization
        vis = img.copy()
        cv.drawChessboardCorners(vis, board_size, corners_refined, found)
        outname = os.path.join('images', 'detected_' + os.path.basename(fname))
        cv.imwrite(outname, vis)
    else:
        print(f"Chessboard not found in {fname}")

# Validate we detected enough views
if len(objpoints) < 3:
    raise SystemExit("Not enough successful detections for calibration. Need at least 3 good images.")

# Calibrate camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print(f"RMS re-projection error: {ret:.4f}")
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs.ravel())

# Save calibration
np.savez('camera_calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
         rvecs=rvecs, tvecs=tvecs, rms=ret)

# Demonstrate undistortion on the first usable image
sample_img = cv.imread(image_files[0])
h, w = sample_img.shape[:2]
new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
undistorted = cv.undistort(sample_img, camera_matrix, dist_coeffs, None, new_camera_mtx)

# Convert BGR->RGB for matplotlib display
sample_rgb = cv.cvtColor(sample_img, cv.COLOR_BGR2RGB)
undist_rgb = cv.cvtColor(undistorted, cv.COLOR_BGR2RGB)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(sample_rgb); axs[0].set_title('Original'); axs[0].axis('off')
axs[1].imshow(undist_rgb); axs[1].set_title('Undistorted'); axs[1].axis('off')
plt.tight_layout()
plt.show()

# If you want per-image reprojection error (useful for diagnostics):
mean_errors = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_errors.append(error)
print(f"Mean reprojection error per image: min {min(mean_errors):.4f}, max {max(mean_errors):.4f}, mean {np.mean(mean_errors):.4f}")
