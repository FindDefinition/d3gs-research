
import numpy as np 
def get_rotation_matrix(roll, pitch, yaw, name=None):
  """Gets a rotation matrix given roll, pitch, yaw.

  roll-pitch-yaw is z-y'-x'' intrinsic rotation which means we need to apply
  x(roll) rotation first, then y(pitch) rotation, then z(yaw) rotation.

  https://en.wikipedia.org/wiki/Euler_angles
  http://planning.cs.uiuc.edu/node102.html

  Args:
    roll : x-rotation in radians.
    pitch: y-rotation in radians. The shape must be the same as roll.
    yaw: z-rotation in radians. The shape must be the same as roll.
    name: the op name.

  Returns:
    A rotation tensor with the same data type of the input. Its shape is
      [input_shape_of_yaw, 3 ,3].
  """
  with tf.compat.v1.name_scope(name, 'GetRotationMatrix', [yaw, pitch, roll]):
    cos_roll = tf.cos(roll)
    sin_roll = tf.sin(roll)
    cos_yaw = tf.cos(yaw)
    sin_yaw = tf.sin(yaw)
    cos_pitch = tf.cos(pitch)
    sin_pitch = tf.sin(pitch)

    ones = tf.ones_like(yaw)
    zeros = tf.zeros_like(yaw)

    r_roll = tf.stack([
        tf.stack([ones, zeros, zeros], axis=-1),
        tf.stack([zeros, cos_roll, -1.0 * sin_roll], axis=-1),
        tf.stack([zeros, sin_roll, cos_roll], axis=-1),
    ],
                      axis=-2)
    r_pitch = tf.stack([
        tf.stack([cos_pitch, zeros, sin_pitch], axis=-1),
        tf.stack([zeros, ones, zeros], axis=-1),
        tf.stack([-1.0 * sin_pitch, zeros, cos_pitch], axis=-1),
    ],
                       axis=-2)
    r_yaw = tf.stack([
        tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        tf.stack([zeros, zeros, ones], axis=-1),
    ],
                     axis=-2)

    return tf.matmul(r_yaw, tf.matmul(r_pitch, r_roll))

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
        np.stack([cos_pitch, zeros, sin_pitch], axis=-1),
        np.stack([zeros, ones, zeros], axis=-1),
        np.stack([-1.0 * sin_pitch, zeros, cos_pitch], axis=-1),
    ],
                       axis=-2)
    r_yaw = np.stack([
        np.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        np.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        np.stack([zeros, zeros, ones], axis=-1),
    ],
                     axis=-2)

    return np.matmul(r_yaw, np.matmul(r_pitch, r_roll))

"""
        using T = float;
        T roll = pose[3];
        T pitch = pose[4];
        T yaw = pose[5];
        T cos_roll = cos(roll), cos_pitch = cos(pitch), cos_yaw = cos(yaw),
            sin_roll = sin(roll), sin_pitch = sin(pitch), sin_yaw = sin(yaw);
        std::array<std::array<T, 4>, 4> trans;
        // Eigen::Matrix4f trans = Eigen::Matrix4f::Zero();
        trans[0, 0] = cos_yaw * cos_roll - sin_yaw * sin_pitch * sin_roll;
        trans[1, 0] = sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll;
        trans[2, 0] = -cos_pitch * sin_roll;
        trans[3, 0] = 0;
        trans[0, 1] = -sin_yaw * cos_pitch;
        trans[1, 1] = cos_yaw * cos_pitch;
        trans[2, 1] = sin_pitch;
        trans[3, 1] = 0;
        trans[0, 2] = cos_yaw * sin_roll + sin_yaw * sin_pitch * cos_roll;
        trans[1, 2] = sin_yaw * sin_roll - cos_yaw * sin_pitch * cos_roll;
        trans[2, 2] = cos_pitch * cos_roll;
        trans[3, 2] = 0;
        trans[0, 3] = pose[0];
        trans[1, 3] = pose[1];
        trans[2, 3] = pose[2];
        trans[3, 3] = T(1);


"""

def get_rotation_matrix_np_another_impl(roll, pitch, yaw):
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)

    trans = np.eye(3)
    trans[0, 0] = cos_yaw * cos_roll - sin_yaw * sin_pitch * sin_roll
    trans[1, 0] = sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll
    trans[2, 0] = -cos_pitch * sin_roll
    trans[0, 1] = -sin_yaw * cos_pitch
    trans[1, 1] = cos_yaw * cos_pitch
    trans[2, 1] = sin_pitch
    trans[0, 2] = cos_yaw * sin_roll + sin_yaw * sin_pitch * cos_roll
    trans[1, 2] = sin_yaw * sin_roll - cos_yaw * sin_pitch * cos_roll
    trans[2, 2] = cos_pitch * cos_roll

    return trans 


def main():
    roll = np.random.uniform(-np.pi, np.pi, size=[]).astype(np.float32)
    pitch = np.random.uniform(-np.pi, np.pi, size=[]).astype(np.float32)
    yaw = np.random.uniform(-np.pi, np.pi, size=[]).astype(np.float32)

    res = get_rotation_matrix_np(roll, pitch, yaw)
    res2 = get_rotation_matrix_np_another_impl(roll, -pitch, yaw)
    print(res)
    print(res2)

if __name__ == "__main__":
    main()


