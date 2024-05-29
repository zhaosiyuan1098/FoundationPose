from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2


#

model = load_model("/home/siyuan/code/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/siyuan/code/Grounded-Segment-Anything/groundingdino_swint_ogc.pth")
IMAGE_PATH = "realsense/data/mouse/rgb/0000049.png"
TEXT_PROMPT = "mouse."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)