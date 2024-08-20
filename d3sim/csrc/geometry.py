    
import pccm 
from cumm.common import TensorViewArrayLinalg, TensorViewNVRTC


class GeometryOps(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewArrayLinalg, TensorViewNVRTC)

    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def swap(self):
        code = pccm.code()
        code.targ("T")
        code.arg("a, b", "TV_METAL_THREAD T&")
        code.raw(f"""
        T c(a); a=b; b=c;
        """)
        return code

    @pccm.static_function(attrs=["TV_HOST_DEVICE_INLINE"], header_only=True)
    def ray_aabb_intersection(self):
        code = pccm.code()
        code.arg("pos, dir, aabb_min, aabb_max", "const TV_METAL_THREAD tv::array<float, 3>&")
        code.raw("""
        float tmin = (aabb_min[0] - pos[0]) / dir[0];
        float tmax = (aabb_max[0] - pos[0]) / dir[0];
        if (tmin > tmax) {
            swap(tmin, tmax);
        }
        float tymin = (aabb_min[1] - pos[1]) / dir[1];
        float tymax = (aabb_max[1] - pos[1]) / dir[1];
        if (tymin > tymax) {
            swap(tymin, tymax);
        }
        if (tmin > tymax || tymin > tmax) {
            return std::make_tuple(tv::array<float, 2>{ std::numeric_limits<float>::max(), std::numeric_limits<float>::max() }, false);
        }
        if (tymin > tmin)
            tmin = tymin;
        if (tymax < tmax)
            tmax = tymax;
        float tzmin = (aabb_min[2] - pos[2]) / dir[2];
        float tzmax = (aabb_max[2] - pos[2]) / dir[2];
        if (tzmin > tzmax) {
            swap(tzmin, tzmax);
        }
        if (tmin > tzmax || tzmin > tmax) {
            return std::make_tuple(tv::array<float, 2>{ std::numeric_limits<float>::max(), std::numeric_limits<float>::max() }, false);
        }
        if (tzmin > tmin) {
            tmin = tzmin;
        }
        if (tzmax < tmax) {
            tmax = tzmax;
        }
        return std::make_tuple(tv::array<float, 2>{ tmin, tmax }, true);
        """)
        return code.ret("std::tuple<tv::array<float, 2>, bool>")
