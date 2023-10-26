import torch
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder

from ..builder import ROTATED_BBOX_CODERS
from ..transforms import obb2poly, poly2obb


@ROTATED_BBOX_CODERS.register_module()
class DeltaTLTRCECoder(BaseBBoxCoder):
    """Landms(Topleft, Topright, center) coder.

    this coder encodes landms(x_tl, y_tl, x_tr, y_tr, cx, cy) into delta (dx_tl, dy_tl, dx_tr, dy_tr, dcx, dcy) and
    decodes delta (dx_tl, dy_tl, dx_tr, dy_tr, dcx, dcy) back to original landms (x_tl, y_tl, x_tr, y_tr, cx, cy).

    Args:
        angle_range (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self, **kwargs):

        super(DeltaTLTRCECoder, self).__init__(**kwargs)

    def encode(self, bboxes, gt_landms):
        """Get landm regression transformation deltas.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_landms (torch.Tensor): Target of the transformation, e.g.,
                ground-truth landms.

        Returns:
            torch.Tensor: Landm transformation deltas
        """
        assert bboxes.size(1) == 5
        # print('encode-bboxes: ', bboxes)
        # print('encode-bboxes: ', bboxes.shape)
        # 得到的poly是水平anchor的左上，右上，右下，左下四点
        polys = obb2poly(bboxes, 'oc')
        # print('encode-polys: ', polys)
        # print('encode-polys.shape: ', polys.shape)
        # 将poly的左上和右下与anchor的中心点拼接，得到左上，右上，中心的poly
        polys_ = polys[:, :4]
        center = bboxes[:, :2]
        landms = torch.cat((polys_, center), 1)
        bboxes_w_h = torch.cat((bboxes[:, 2:4], bboxes[:, 2:4], bboxes[:, 2:4]), 1)
        # print('bboxes_w_h: ', bboxes_w_h.shape)
        # print('bboxes_w_h: ', bboxes_w_h)
        # print('gt_landms: ', gt_landms)
        # print('encode-landms: ', landms)
        # indx = torch.where(landms < 0)
        # bboxes_w_h[indx] += (landms[indx] -1)
        # landms[indx] = 1
        # print('new-bboxes_w_h: ', bboxes_w_h)
        
        # landms[indx] = 3
        # print('encode-landms-new: ', landms[del_indx])
        landms_detla = gt_landms - landms
        landms_detla /= bboxes_w_h
        # print('landms_detla: ', landms_detla)
        # temp = bboxes[:, 0].unsqueeze(1).expand(bboxes.size(0), 4).unsqueeze(2)
        # print('temp-shape: ', temp.shape)
        return landms_detla

    def decode(self, bboxes, landm_deltas):
        """Apply transformation `fix_deltas` to `boxes`.

        Args:
            hbboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            fix_deltas (torch.Tensor): Encoded offsets with respect to each \
                roi. Has shape (B, N, num_classes * 4) or (B, N, 4) or \
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H \
               when rois is a grid of anchors.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        polys = obb2poly(bboxes, 'oc')
        polys_ = polys[:, :4]
        center = bboxes[:, :2]
        bboxes_w_h = torch.cat((bboxes[:, 2:4], bboxes[:, 2:4], bboxes[:, 2:4]), 1)
        landms = torch.cat((polys_, center), 1)
        landm_deltas = landm_deltas * bboxes_w_h
        pred_landms = landms + landm_deltas

        return pred_landms