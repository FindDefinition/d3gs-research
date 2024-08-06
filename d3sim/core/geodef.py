import enum 

class EulerIntrinsicOrder(enum.IntEnum):
    """Intrinsic Euler angle order.
    if use single-axis rot matrix Rx, Ry, Rz, 
    to get rot matrix R = Rz @ Ry @ Rx, the
    extrinsic order is XYZ, intrinsic order is ZYX.
    """
    XYZ = 0
    XZY = 1
    YXZ = 2
    YZX = 3
    ZXY = 4
    ZYX = 5