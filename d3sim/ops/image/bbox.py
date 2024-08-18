import time
from d3sim.csrc.inliner import INLINER 
import torch 
import numpy as np 
import pccm 
from tensorpc.flow import observe_function
from cumm.inliner import torch_tensor_to_tv
from cumm import tensorview as tv 

@observe_function
def create_image_bbox_mask(bbox_xywh: torch.Tensor, image_size_wh: tuple[int, int], bbox_pad: np.ndarray | None):
    res = torch.empty([image_size_wh[1], image_size_wh[0]], dtype=torch.uint8, device=bbox_xywh.device)
    width = image_size_wh[0]
    height = image_size_wh[1]
    num_bboxes = bbox_xywh.shape[0]
    assert bbox_xywh.shape[1] == 4 and bbox_xywh.dtype == torch.float32
    if bbox_pad is None:
        bbox_pad = np.zeros((4,), np.float32)
    else:
        if bbox_pad.dtype != np.float32:
            bbox_pad = bbox_pad.astype(np.float32)
    INLINER.kernel_1d(
        f"create_image_bbox_mask", height * width, 0, f"""
    namespace op = tv::arrayops;
    using math_op_t = op::MathScalarOp<float>;
    int pixel_y = i / $width;
    int pixel_x = i % $width;
    bool inside = false;
    auto box_ptr = op::reinterpret_cast_array_nd<4>($bbox_xywh);
    auto box_pad = $bbox_pad;
    for (int j = 0; j < $num_bboxes; ++j){{
        auto box = box_ptr[j];
        inside |= (pixel_x >= (box[0] - box_pad[0]) 
                && pixel_y >= (box[1] - box_pad[1])
                && pixel_x < (box[0] + box[2] + box_pad[2]) 
                && pixel_y < (box[1] + box[3] + box_pad[3]));
    }}
    $res[i] = inside;
    """)
    return res > 0
