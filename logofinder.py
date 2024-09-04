from groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv
from torchvision.ops import box_convert
import cv2 as cv
from PIL import Image
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import os
from torch import Tensor
from typing import Tuple

HOME = os.getcwd()
CONFIG_PATH = os.path.join(
    HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "GroundingDINO", "weights", WEIGHTS_NAME)


def detect_logos(
    gd_model=None,
    img_path: str = None,
    text_prompt: str = "logo",
    box_threshold: float = 0.35,
    text_threshold: float = 1,
) -> Tuple[Tensor]:
    """
    Use grounding dino to find logos/insignia and return bounding boxes etc,
    and plot detections

    Inputs:
        gd_model - grounding dino model
        img_path - img to detect logos in
        text_prompt - prompt for grounding dino
        box_threshold - minimum prediction score for grounding dino results
        text_threshold - minimum similarity of returned object captions to prompt
    Returns:
        boxes, scores, captions - grounding dino results
    """
    image_source, image = load_image(img_path)
    boxes, scores, captions = predict(
        model=gd_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=scores, phrases=captions
    )
    sv.plot_image(annotated_frame, (16, 16))
    return boxes, scores, captions


def sub_image_matcher(
    ref_img: npt.NDArray = None, img: npt.NDArray = None, min_match_count: int = 10
)-> None:
    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(ref_img, None)
    kp2, des2 = sift.detectAndCompute(img, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > min_match_count:
        print("Matches found - {}/{}".format(len(good), min_match_count))
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = ref_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv.perspectiveTransform(pts, M)
        img = cv.polylines(img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        line_color = (0, 255, 0)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), min_match_count))
        matchesMask = None
        line_color = (255, 0, 0)

    draw_params = dict(
        matchColor=line_color,
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    plot_img = cv.drawMatches(ref_img, kp1, img, kp2, good, None, **draw_params)
    plt.imshow(plot_img, "gray")
    plt.axis("off")
    plt.show()


def bb_to_pxls(bb: npt.NDArray = None, i_w: int = None, i_h: int = None):
    """
    Convert bounding box coordinates from normalised coords to pixel coords
    """
    bb_out = np.zeros_like(bb)
    bb_out[0] = int(np.round(bb[0] * i_w))
    bb_out[1] = int(np.round(bb[1] * i_h))
    bb_out[2] = int(np.round(bb[2] * i_w))
    bb_out[3] = int(np.round(bb[3] * i_h))
    return bb_out


def compare_logos(ref_img_path: str = None, img_path: str = None, boxes=None) -> None:
    ref = cv.imread(ref_img_path, cv.IMREAD_GRAYSCALE)
    ref = cv.normalize(ref, None, 0, 255, cv.NORM_MINMAX).astype("uint8")
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype("uint8")
    i_h, i_w = img.shape
    source_image = Image.fromarray(img)
    boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
    for box in boxes_xyxy:
        pxl_box = bb_to_pxls(box.tolist(), i_w, i_h)
        sub_img = source_image.crop(pxl_box)
        sub_img = np.array(sub_img)
        sub_image_matcher(ref_img=ref, img=sub_img)


def load_gd():
    """
    Helper function to load & return Grounding Dino model
    """
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    return model


def find_in_image(
    image_path: str = None,
    reference_path: str = None,
    gd_model=None,
    gd_threshold: float = 0.3,
) -> None:
    """
    Wrapper function to detect logos and attempt to match them to reference image

    Inputs
        image_path - path to image to detect & match logos from
        reference_path - path to reference image of logo
        gd_model - grounding dino model
        gd_threshold - minimum confidence score for GD predictions
    """
    boxes, _, _ = detect_logos(
        gd_model=gd_model, img_path=image_path, box_threshold=gd_threshold
    )
    compare_logos(reference_path, image_path, boxes)
