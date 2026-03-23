# Assignment 1 - Image Warping

### 1. Basic Image Geometric Transformation (Scale/Rotation/Translation).
Fill the [Missing Part](run_global_transform.py#L21) of 'run_global_transform.py'.
### code
```python
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    h, w = image.shape[:2]

    cx = w / 2
    cy = h / 2
    scale_matrix = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])
    theta = np.deg2rad(rotation)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    translation_matrix = np.array([
        [1, 0, translation_x],
        [0, 1, translation_y],
        [0, 0, 1]
    ])
    move_to_origin = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ])
    move_back = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ])
    transform_matrix = (
        translation_matrix
        @ move_back
        @ rotation_matrix
        @ scale_matrix
        @ move_to_origin
    )
    if flip_horizontal:
        flip_matrix = np.array([
            [-1, 0, w],
            [0, 1, 0],
            [0, 0, 1]
        ])
        transform_matrix = flip_matrix @ transform_matrix
        
    affine_matrix = transform_matrix[:2, :]

    transformed_image = cv2.warpAffine(
        image,
        affine_matrix,
        (w, h),
        borderValue=(255,255,255)
    )
    
    return transformed_image
```

### 2. Point Based Image Deformation.
### code
```python
Implement MLS or RBF based image deformation in the [Missing Part](run_point_transform.py#L52) of 'run_point_transform.py'.

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """
    warped_image = np.array(image)
    ### FILL: Implement MLS or RBF based image warping
    h, w = image.shape[:2]
    displacement = target_pts - source_pts
    warped_image = np.zeros_like(image)
    if len(source_pts) == 0 or len(target_pts) == 0:
        return image
    for y in range(h):
        for x in range(w):
            p = np.array([x, y])
            weights = []
            disp = np.zeros(2)
            #每个点操作
            for i in range(len(source_pts)):
                r = np.linalg.norm(p - source_pts[i])   #距离
                w_i = 1.0 / ((r ** alpha) + eps)        #权重
                weights.append(w_i)
                disp += w_i * displacement[i]           #位移叠加
            weights = np.array(weights)
            #归一化
            if np.sum(weights) > 0:
                disp = disp / np.sum(weights)
            #变形后的坐标
            new_pos = p + disp
            new_x = int(np.clip(new_pos[0], 0, w - 1))
            new_y = int(np.clip(new_pos[1], 0, h - 1))
            warped_image[y, x] = image[new_y, new_x]
    return warped_image
```


## Implementation of Image Geometric Transformation

This repository is Xiaogang Tang's implementation of Assignment_01 of DIP. 



## Running

To run basic transformation, run:

```basic
python run_global_transform.py
```

To run point guided transformation, run:

```point
python run_point_transform.py
```

## Results
### Basic Transformation
<img src="pics/basic transformation1.png" alt="alt text" width="800">
<img src="pics/basic transformation2.png" alt="alt text" width="800">

### Point Guided Deformation:
<img src="pics/point guided transformation1.png" alt="alt text" width="800">
<img src="pics/point guided transformation2.png" alt="alt text" width="800">

## Acknowledgement

>📋 Thanks for the algorithms proposed by [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf).
