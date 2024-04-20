import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os

sam_checkpoint = "/home/siyuan/code/segment-anything/sam_vit_h_4b8939.pth"
masks_dir="data/realsense/masks"
os.makedirs(masks_dir, exist_ok=True)



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

image = cv2.imread('data/realsense/rgb/272.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()


from segment_anything import sam_model_registry, SamPredictor

model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

predictor.set_image(image)

input_point = np.array([[300,200]])
input_label = np.array([1])


# plt.figure(figsize=(10,10))
# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.show()  

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

print(masks.shape)  # (number_of_masks) x H x W

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()
    key = cv2.waitKey(0)
    # if key == ord('s'):
        # mask_image = (mask * 255).astype(np.uint8)
        # cv2.imwrite(os.path.join(masks_dir, f"mask_{i+1}.png"), mask_image)
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    _, binary_image = cv2.threshold(binary_mask, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binary Image", binary_image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord('s'):
        cv2.imwrite(os.path.join(masks_dir, '{}.png'.format(i)), binary_mask)

# for i, mask in enumerate(masks):
#     mask_image = (mask * 255).astype(np.uint8)
#     cv2.imwrite(f"mask_{i+1}.png", mask_image)

  