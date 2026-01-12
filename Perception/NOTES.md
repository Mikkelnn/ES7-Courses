## f-stop

1 stop = x**2
f/4 → f/8 (−2 stops) and exposure time 4 ms → 1 ms (−2 stops) = (-4)**2 = 16x lighting needed

## Inverse square law

1000 lux at 20 cm what is the illuminance at 40 cm -> How many f-stops is needed to compensate?
E ∝ 1/(d^2); E = intensity; ∝ = at; d = disrance

E40​=1000×(20/40)\*\*2 = 250 lux. That’s −2 stops (4× less light).

## Glossary

- Peak signal-to-noise ratio (PSNR)
- Structural Similarity Index Measure (SSIM)

- Low Recall = meny False-Negatives
- High precition = few False-Positive

## Confusion matrix

Precition = TP/(FP + TP)
Accuracy (TP + TN)/(TP + TN + FP + FN)

## Dynamic range

dB = 20 \* log_10(2^(bit-res))

## 2D transforms

- Rigid = rotation + translation (no scaling)
- Similarity = rigid + uniform scaling
- Affine = similarity + shear & non-uniform scaling
- Projective = most general; models perspective effects (camera view)

| Property Preserved | Rigid (Euclidean) | Similarity | Affine | Projective |
| ------------------ | ----------------- | ---------- | ------ | ---------- |
| Parallel Lines     | Yes               | Yes        | Yes    | No         |
| Ratio of Lengths   | Yes               | Yes        | No     | No         |
| Angles             | Yes               | Yes        | No     | No         |

| Property                     | Rigid (Euclidean) | Similarity        | Affine | Projective (Homography) |
| ---------------------------- | ----------------- | ----------------- | ------ | ----------------------- |
| Translation                  | Yes               | Yes               | Yes    | Yes                     |
| Rotation                     | Yes               | Yes               | Yes    | No                      |
| Uniform Scaling              | No                | Yes               | No     | No                      |
| Non-uniform Scaling          | No                | No                | Yes    | Yes                     |
| Shear                        | No                | No                | Yes    | Yes                     |
| Perspective Distortion       | No                | No                | No     | Yes                     |
| Distance Preservation        | Yes               | No                | No     | No                      |
| Angle Preservation           | Yes               | Yes               | No     | No                      |
| Parallel Lines Preserved     | Yes               | Yes               | Yes    | No                      |
| Collinearity Preserved       | Yes               | Yes               | Yes    | Yes                     |
| Ratio of Lengths (same line) | Yes               | Yes               | No     | No                      |
| Cross-ratio Preserved        | No                | No                | No     | Yes                     |
| Shape Preservation           | Yes               | Yes (up to scale) | No     | No                      |
| Degrees of Freedom (2D)      | 3                 | 4                 | 6      | 8                       |
| Matrix Size                  | 3x3               | 3x3               | 3x3    | 3x3                     |
