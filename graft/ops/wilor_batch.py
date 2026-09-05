"""Batched WiLoR hand-pose refinement.

Stock WiLorHandPose3dEstimationPipeline.predict_with_bboxes already batches all hands
WITHIN one image into a single wilor_model forward; the only B=1 tie is that it takes a
single image. batched_predict_with_bboxes(pipeline, images, bboxes_list, is_rights_list)
crops hands from all B images (verbatim stock crop math), concatenates into ONE
(sum_N,256,256,3) tensor, runs wilor_model ONCE, and scatters outputs back per (image,hand).
Crop + post-processing are copied verbatim from the stock pipeline, so the only numerical
difference vs per-image is the fp16 cuBLAS reduction order in the single forward
(verified: hand_pose max ~2e-3 rad / 1 fp16-ULP; ~1.5-2.3x speedup at B=2-8).
"""
import cv2
import numpy as np
import torch
from skimage.filters import gaussian
from wilor_mini.utils import utils


def _build_patches_for_image(pipeline, image, bboxes, is_rights, rescale_factor=2.5):
    """Replicates the crop-building loop of predict_with_bboxes for ONE image.

    Returns
    -------
    patches : np.ndarray (N, 256, 256, 3) float
    meta    : list of dicts with per-hand {center, bbox_size, right, img_size, bbox4}
    """
    if len(bboxes) == 0:
        return np.empty((0, pipeline.IMAGE_SIZE, pipeline.IMAGE_SIZE, 3), dtype=np.float32), []

    bboxes = np.asarray(bboxes, dtype=np.float32)
    center = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2.0
    scale = rescale_factor * (bboxes[:, 2:4] - bboxes[:, 0:2])

    patches = []
    meta = []
    for i in range(bboxes.shape[0]):
        bbox_size = scale[i].max()
        patch_width = patch_height = pipeline.IMAGE_SIZE
        right = is_rights[i]
        flip = right == 0
        box_center = center[i]

        cvimg = image.copy()
        downsampling_factor = ((bbox_size * 1.0) / patch_width)
        downsampling_factor = downsampling_factor / 2.0
        if downsampling_factor > 1.1:
            cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2,
                             channel_axis=2, preserve_range=True)
        img_size = np.array([cvimg.shape[1], cvimg.shape[0]])

        img_patch_cv, _ = utils.generate_image_patch_cv2(
            cvimg, box_center[0], box_center[1], bbox_size, bbox_size,
            patch_width, patch_height, flip, 1.0, 0,
            border_mode=cv2.BORDER_CONSTANT)
        patches.append(img_patch_cv)
        meta.append({
            'center': box_center,
            'bbox_size': bbox_size,
            'right': right,
            'img_size': img_size,
            'bbox4': bboxes[i, :4].tolist(),
        })

    patches = np.stack(patches)
    return patches, meta


def _postprocess_hand(pipeline, wilor_output_i, m):
    """Replicates the per-hand post-processing of predict_with_bboxes.

    wilor_output_i : dict of np arrays already sliced to this hand (leading dim 1).
    m              : metadata dict from _build_patches_for_image.
    """
    pred_cam = wilor_output_i["pred_cam"]
    bbox_size = m['bbox_size']
    box_center = m['center']
    right = m['right']
    img_size = m['img_size']

    multiplier = (2 * right - 1)
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    if right == 0:
        wilor_output_i["pred_keypoints_3d"][:, :, 0] = -wilor_output_i["pred_keypoints_3d"][:, :, 0]
        wilor_output_i["pred_vertices"][:, :, 0] = -wilor_output_i["pred_vertices"][:, :, 0]
        wilor_output_i["global_orient"] = np.concatenate(
            (wilor_output_i["global_orient"][:, :, 0:1], -wilor_output_i["global_orient"][:, :, 1:3]),
            axis=-1)
        wilor_output_i["hand_pose"] = np.concatenate(
            (wilor_output_i["hand_pose"][:, :, 0:1], -wilor_output_i["hand_pose"][:, :, 1:3]),
            axis=-1)
    scaled_focal_length = pipeline.FOCAL_LENGTH / pipeline.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = utils.cam_crop_to_full(pred_cam, box_center[None], bbox_size,
                                             img_size[None], scaled_focal_length)
    wilor_output_i["pred_cam_t_full"] = pred_cam_t_full
    wilor_output_i["scaled_focal_length"] = scaled_focal_length
    pred_keypoints_2d = utils.perspective_projection(
        wilor_output_i["pred_keypoints_3d"], translation=pred_cam_t_full,
        focal_length=np.array([scaled_focal_length] * 2)[None],
        camera_center=img_size[None] / 2)
    wilor_output_i["pred_keypoints_2d"] = pred_keypoints_2d
    return wilor_output_i


@torch.no_grad()
def batched_predict_with_bboxes(pipeline, images, bboxes_list, is_rights_list,
                                rescale_factor=2.5):
    """Process hands from B images in ONE wilor_model forward.

    Parameters
    ----------
    pipeline      : WiLorHandPose3dEstimationPipeline
    images        : list of B HxWx3 uint8/float numpy images
    bboxes_list   : list of B arrays (N_b, 4)  (x1,y1,x2,y2)
    is_rights_list: list of B sequences of {0,1} of length N_b

    Returns
    -------
    list of B lists of detect_ret dicts (same structure as predict_with_bboxes output).
    """
    all_patches = []
    per_image_meta = []
    counts = []
    for image, bboxes, is_rights in zip(images, bboxes_list, is_rights_list):
        patches, meta = _build_patches_for_image(pipeline, image, bboxes, is_rights,
                                                  rescale_factor=rescale_factor)
        all_patches.append(patches)
        per_image_meta.append(meta)
        counts.append(len(meta))

    total = sum(counts)
    results = [[] for _ in range(len(images))]
    if total == 0:
        return results

    big = np.concatenate([p for p in all_patches if p.shape[0] > 0], axis=0)
    big_t = torch.from_numpy(big).to(device=pipeline.device, dtype=pipeline.dtype)
    wilor_output = pipeline.wilor_model(big_t)
    wilor_output = {k: v.cpu().float().numpy() for k, v in wilor_output.items()}

    flat = 0
    for b, meta in enumerate(per_image_meta):
        for m in meta:
            wilor_output_i = {key: val[[flat]] for key, val in wilor_output.items()}
            wilor_output_i = _postprocess_hand(pipeline, wilor_output_i, m)
            results[b].append({
                'hand_bbox': m['bbox4'],
                'is_right': m['right'],
                'wilor_preds': wilor_output_i,
            })
            flat += 1
    return results
