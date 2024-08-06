
import numpy as np 
from d3sim.core.geodef import EulerIntrinsicOrder

def euler_to_rotmat_np(roll, pitch, yaw, order: EulerIntrinsicOrder):
    """Converts Euler angles to rotation matrix.

    Args:
      roll: x-rotation in radians.
      pitch: y-rotation in radians.
      yaw: z-rotation in radians.
      order: EulerIntrinsicOrder, the rotation order.

    Returns:
      A rotation tensor with the same data type of the input. Its shape is
        [input_shape_of_yaw, 3 ,3].
    """
    a = np.cos(roll)
    b = np.sin(roll)
    c = np.cos(pitch)
    d = np.sin(pitch)
    e = np.cos(yaw)
    f = np.sin(yaw)

    if order == EulerIntrinsicOrder.XYZ:
        ae = a * e
        af = a * f
        be = b * e
        bf = b * f

        return np.array([
            [c * e, -c * f, d],
            [af + be * d, ae - bf * d, -b * c],
            [bf - ae * d, be + af * d, a * c],
        ])
    elif order == EulerIntrinsicOrder.YXZ:
        ce = c * e
        cf = c * f
        de = d * e
        df = d * f

        return np.array([
            [ce + df * b, de * b - cf, a * d],
            [a * f, a * e, -b],
            [cf * b - de, df + ce * b, a * c],
        ])
    elif order == EulerIntrinsicOrder.ZXY:
        ce = c * e
        cf = c * f
        de = d * e
        df = d * f

        return np.array([
            [ce - df * b, -a * f, de + cf * b],
            [cf + de * b, a * e, df - ce * b],
            [-a * d, b, a * c],
        ])
    elif order == EulerIntrinsicOrder.ZYX:
        ae = a * e
        af = a * f
        be = b * e
        bf = b * f
        return np.array([
            [c * e, be * d - af, ae * d + bf],
            [c * f, bf * d + ae, af * d - be],
            [-d, b * c, a * c],
        ])
    elif order == EulerIntrinsicOrder.YZX:
        ac = a * c
        ad = a * d
        bc = b * c
        bd = b * d

        return np.array([
            [c * e, bd - ac * f, bc * f + ad],
            [f, a * e, -b * e],
            [-d * e, ad * f + bc, ac - bd * f],
        ])
    elif order == EulerIntrinsicOrder.XZY:
        ac = a * c
        ad = a * d
        bc = b * c
        bd = b * d

        return np.array([
            [c * e, -f, d * e],
            [ac * f + bd, a * e, ad * f - bc],
            [bc * f - ad, b * e, bd * f + ac],
        ])
    else:
        raise ValueError("Unsupported Euler rotation order.")

def euler_from_matrix_np(matrix: np.ndarray, order: EulerIntrinsicOrder):
    """Converts rotation matrix to Euler angles.

    Args:
      matrix: A rotation tensor with the same data type of the input. Its shape
        is [3 ,3].
      order: EulerIntrinsicOrder, the rotation order (intrinsic).

    Returns:
      A tuple of roll, pitch, yaw in radians. The shape is the same as the input
        matrix.
    """
    m11 = matrix[0, 0]
    m12 = matrix[0, 1]
    m13 = matrix[0, 2]
    m21 = matrix[1, 0]
    m22 = matrix[1, 1]
    m23 = matrix[1, 2]
    m31 = matrix[2, 0]
    m32 = matrix[2, 1]
    m33 = matrix[2, 2]

    if order == EulerIntrinsicOrder.XYZ:
        pitch = np.arcsin(np.clip(m13, -1, 1))
        if np.abs(m13) < 0.9999999:
            roll = np.arctan2(-m23, m33)
            yaw = np.arctan2(-m12, m11)
        else:
            roll = np.arctan2(m32, m22)
            yaw = 0
    elif order == EulerIntrinsicOrder.YXZ:
        roll = np.arcsin(-np.clip(m23, -1, 1))
        if np.abs(m23) < 0.9999999:
            pitch = np.arctan2(m13, m33)
            yaw = np.arctan2(m21, m22)
        else:
            pitch = np.arctan2(-m31, m11)
            yaw = 0
    elif order == EulerIntrinsicOrder.ZXY:
        roll = np.arcsin(np.clip(m32, -1, 1))
        if np.abs(m32) < 0.9999999:
            pitch = np.arctan2(-m31, m33)
            yaw = np.arctan2(-m12, m22)
        else:
            pitch = 0
            yaw = np.arctan2(m21, m11)

    elif order == EulerIntrinsicOrder.ZYX:
        pitch = np.arcsin(-np.clip(m31, -1, 1))
        if np.abs(m31) < 0.9999999:
            roll = np.arctan2(m32, m33)
            yaw = np.arctan2(m21, m11)
        else:
            roll = 0
            yaw = np.arctan2(-m12, m22)
    elif order == EulerIntrinsicOrder.YZX:
        yaw = np.arcsin(np.clip(m21, -1, 1))
        if np.abs(m21) < 0.9999999:
            roll = np.arctan2(-m23, m22)
            pitch = np.arctan2(-m31, m11)
        else:
            roll = 0
            pitch = np.arctan2(m13, m33)
    elif order == EulerIntrinsicOrder.XZY:
        yaw = np.arcsin(-np.clip(m12, -1, 1))
        if np.abs(m12) < 0.9999999:
            roll = np.arctan2(m32, m22)
            pitch = np.arctan2(m13, m11)
        else:
            roll = np.arctan2(-m23, m33)
            pitch = 0
    else:
        raise ValueError("Unsupported Euler rotation order.")
    
    return roll, pitch, yaw
    

def get_rotation_matrix_np(roll, pitch, yaw):
    """Gets a rotation matrix given roll, pitch, yaw.

    roll-pitch-yaw is z-y'-x'' intrinsic rotation which means we need to apply
    x(roll) rotation first, then y(pitch) rotation, then z(yaw) rotation.

    https://en.wikipedia.org/wiki/Euler_angles
    http://planning.cs.uiuc.edu/node102.html

    Args:
      roll : x-rotation in radians.
      pitch: y-rotation in radians. The shape must be the same as roll.
      yaw: z-rotation in radians. The shape must be the same as roll.

    Returns:
      A rotation tensor with the same data type of the input. Its shape is
        [input_shape_of_yaw, 3 ,3].
    """
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)

    ones = np.ones_like(yaw)
    zeros = np.zeros_like(yaw)

    r_roll = np.stack([
        np.stack([ones, zeros, zeros], axis=-1),
        np.stack([zeros, cos_roll, -1.0 * sin_roll], axis=-1),
        np.stack([zeros, sin_roll, cos_roll], axis=-1),
    ],
                      axis=-2)
    r_pitch = np.stack([
        np.stack([cos_pitch, zeros, 1.0 * sin_pitch], axis=-1),
        np.stack([zeros, ones, zeros], axis=-1),
        np.stack([-sin_pitch, zeros, cos_pitch], axis=-1),
    ],
                       axis=-2)
    r_yaw = np.stack([
        np.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        np.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        np.stack([zeros, zeros, ones], axis=-1),
    ],
                     axis=-2)

    return np.matmul(r_yaw, np.matmul(r_pitch, r_roll))


def __main():
    from scipy.spatial.transform import Rotation
    random_euler = np.random.rand(3)
    roll, pitch, yaw = random_euler
    rot = Rotation.from_euler("zyx", random_euler).as_matrix()
    rot_my_2 = euler_to_rotmat_np(roll, pitch, yaw, EulerIntrinsicOrder.ZYX)
    euler_my = euler_from_matrix_np(rot_my_2, EulerIntrinsicOrder.ZYX)
    print(euler_my, random_euler)
    # rot_my = get_rotation_matrix_np(roll, pitch, yaw)
    # print(rot_my_2, )
    # print(rot_my)
    # assert np.allclose(rot_my, rot_my_2)
    # rot_my_2 = euler_to_rotmat_np(0, 0, 1, EulerIntrinsicOrder.XYZ)
    # print(rot_my_2)
    pass 

if __name__ == "__main__":
    __main()