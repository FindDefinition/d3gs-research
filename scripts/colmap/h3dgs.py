import hashlib
import tempfile
from d3sim.algos.d3gs.hd3gs.data.chunks import ColmapSceneSplitChunks
from d3sim.data.scene_def import Scene 
from pathlib import Path 
from d3sim.data.transforms.colmap import ColmapBundleAdjustment, ColmapCopySubsetImageFromModel, ColmapCreatePriorFromModel, ColmapFetchResult, ColmapMetaHandleBase, ColmapPointTriangulator, ColmapPriorModelCustomMatchKNN, ColmapTBALoop, ColmapToRefUnrectifiedResult, ColmapWorkDirs, ColmapFeatureExtract, ColmapCustomMatchGen, ColmapCustomMatch, ColmapMapper, ColmapUndistort, ColmapFilterFloatingAndNoSFM, ColmapRefineRotAndScale, ColmapFromRefUnrectifiedResult, align_two_colmap_models


def __main_test_simpify():
    img_bin_folder = "/root/autodl-tmp/example_dataset/camera_calibration/unrectified"
    fake_scene = Scene("wtf", [])
    fake_scene.uri = img_bin_folder
    fake_scene.set_user_data("model_output", img_bin_folder)
    ColmapMetaHandleBase.store_colmap_model_path(fake_scene, img_bin_folder)

    workdirs = ColmapWorkDirs(Path(img_bin_folder) / "__colmap", create=True)
    fake_scene = ColmapFilterFloatingAndNoSFM()(fake_scene)
    model_output = Path(ColmapMetaHandleBase.get_colmap_meta_from_scene_required(fake_scene).colmap_model_path)
    images_bin = Path(model_output) / "images.bin"
    ref_images_bin = Path(img_bin_folder) / "sparse/0" / "images.bin"
    print(images_bin, ref_images_bin)
    md5_images_bin = hashlib.md5(images_bin.read_bytes()).hexdigest()
    md5_ref_images_bin = hashlib.md5(ref_images_bin.read_bytes()).hexdigest()
    print(md5_images_bin, md5_ref_images_bin)
    assert md5_images_bin == md5_ref_images_bin

def __main_test_refine():
    img_bin_folder = "/root/autodl-tmp/example_dataset/camera_calibration/rectified"
    fake_scene = Scene("wtf", [])
    fake_scene.uri = img_bin_folder
    ColmapMetaHandleBase.store_colmap_model_path(fake_scene, str(Path(img_bin_folder) / "sparse"))
    workdirs = ColmapWorkDirs(Path(img_bin_folder) / "__colmap", create=True)
    fake_scene = ColmapRefineRotAndScale()(fake_scene)
    model_output = Path(ColmapMetaHandleBase.get_colmap_meta_from_scene_required(fake_scene).colmap_model_path)
    images_bin = Path(model_output) / "images.bin"
    ref_images_bin = Path("/root/autodl-tmp/example_dataset/camera_calibration/aligned") / "sparse/0" / "images.bin"
    print(images_bin, ref_images_bin)
    md5_images_bin = hashlib.md5(images_bin.read_bytes()).hexdigest()
    md5_ref_images_bin = hashlib.md5(ref_images_bin.read_bytes()).hexdigest()
    print(md5_images_bin, md5_ref_images_bin)
    assert md5_images_bin == md5_ref_images_bin

def __main_test_post_model():
    img_bin_folder = "/root/autodl-tmp/example_dataset/camera_calibration/rectified"
    fake_scene = Scene("wtf", [])
    fake_scene.uri = img_bin_folder
    ColmapMetaHandleBase.store_colmap_model_path(fake_scene, str(Path(img_bin_folder) / "sparse"))
    workdirs = ColmapWorkDirs(Path(img_bin_folder) / "__colmap", create=True)
    ColmapRefineRotAndScale()(fake_scene)

def __main_from_ref_result():
    test_refine: bool = True
    ref_calib_root = Path("/root/autodl-tmp/example_dataset/camera_calibration")
    img_bin_folder = ref_calib_root / "debug_dev"
    from d3sim.algos.d3gs.hd3gs.data.load import _load_debug_scene
    fake_scene = _load_debug_scene()
    fake_scene.uri = str(img_bin_folder)
    ColmapMetaHandleBase.store_colmap_model_path(fake_scene, str(Path(img_bin_folder) / "sparse"))
    workdirs = ColmapWorkDirs(Path(img_bin_folder) / "__colmap", create=True)
    fake_scene = ColmapFromRefUnrectifiedResult(ref_unrectified_result_path=str(ref_calib_root / "unrectified/sparse/0"))(fake_scene)
    
    fake_scene = ColmapFilterFloatingAndNoSFM()(fake_scene)
    if test_refine:
        fake_scene = ColmapRefineRotAndScale()(fake_scene)

    fake_scene = ColmapToRefUnrectifiedResult()(fake_scene)
    model_output = Path(ColmapMetaHandleBase.get_colmap_meta_from_scene_required(fake_scene).colmap_model_path)
    if test_refine:
        ref_images_bin = Path(ref_calib_root) / "aligned" / "sparse/0" / "images.bin"
    else:
        ref_images_bin = Path(ref_calib_root) / "unrectified" / "sparse/0" / "images.bin"
    md5_images_bin = hashlib.md5((model_output / "images.bin").read_bytes()).hexdigest()
    md5_ref_images_bin = hashlib.md5(ref_images_bin.read_bytes()).hexdigest()
    print(md5_images_bin, md5_ref_images_bin)
    assert md5_images_bin == md5_ref_images_bin

def __main_from_ref_mask_result():
    test_filter_only: bool = False
    ref_calib_root = Path("/root/autodl-tmp/example_dataset/camera_calibration")
    img_bin_folder = ref_calib_root / "debug_dev"
    from d3sim.algos.d3gs.hd3gs.data.load import _load_debug_scene
    fake_scene = _load_debug_scene()
    fake_scene.uri = str(img_bin_folder)
    ColmapMetaHandleBase.store_colmap_model_path(fake_scene, str(Path(img_bin_folder) / "sparse"))
    workdirs = ColmapWorkDirs(Path(img_bin_folder) / "__colmap", create=True)
    fake_scene = ColmapFromRefUnrectifiedResult(ref_unrectified_result_path=str(ref_calib_root / "unrectified/sparse/0"))(fake_scene)
    
    fake_scene = ColmapFilterFloatingAndNoSFM()(fake_scene)
    if test_filter_only:
        fake_scene = ColmapToRefUnrectifiedResult()(fake_scene)

        model_output = Path(ColmapMetaHandleBase.get_colmap_meta_from_scene_required(fake_scene).colmap_model_path)
        ref_images_bin = Path(ref_calib_root) / "unrectified" / "sparse/0" / "images.bin"
        md5_images_bin = hashlib.md5((model_output / "images.bin").read_bytes()).hexdigest()
        md5_ref_images_bin = hashlib.md5(ref_images_bin.read_bytes()).hexdigest()
        assert md5_images_bin == md5_ref_images_bin
        print(md5_images_bin, md5_ref_images_bin)

        return 
    fake_scene = ColmapFetchResult()(fake_scene)
    fake_scene = ColmapUndistort()(fake_scene)
    fake_scene = ColmapRefineRotAndScale()(fake_scene)

    # breakpoint()

    fake_scene = ColmapToRefUnrectifiedResult()(fake_scene)
    model_output = Path(ColmapMetaHandleBase.get_colmap_meta_from_scene_required(fake_scene).colmap_model_path)
    print(fake_scene.get_user_data_type_checked("model_output", str))
    ref_images_bin = Path(ref_calib_root) / "aligned" / "sparse/0" / "images.bin"
    md5_images_bin = hashlib.md5((model_output / "images.bin").read_bytes()).hexdigest()
    md5_ref_images_bin = hashlib.md5(ref_images_bin.read_bytes()).hexdigest()
    print(md5_images_bin, md5_ref_images_bin)
    assert md5_images_bin == md5_ref_images_bin

def __test_make_chunk():
    ref_calib_root = Path("/root/autodl-tmp/example_dataset/camera_calibration")
    ref_img_root = ref_calib_root / "rectified" / "images"
    ref_model = Path(ref_calib_root) / "aligned" / "sparse/0"
    fake_scene = Scene("wtf", [])
    fake_scene.uri = str(ref_calib_root / "dev")
    ColmapMetaHandleBase.store_colmap_model_path(fake_scene, str(ref_model))
    ColmapMetaHandleBase.store_colmap_image_root(fake_scene, str(ref_img_root))

    res = ColmapSceneSplitChunks(debug=False)(fake_scene)
    for s in res:
        print(s.uri)
    create_db = ColmapCreatePriorFromModel()
    create_knn_match = ColmapPriorModelCustomMatchKNN(k=200)
    copy_img = ColmapCopySubsetImageFromModel()
    feature_extract = ColmapFeatureExtract(from_existing_model=True)
    custom_match = ColmapCustomMatch()
    tbaloop = ColmapTBALoop(
        num_loop=2,
        point_triangulator=ColmapPointTriangulator(),
        bundle_adjustment=ColmapBundleAdjustment(),
    )
    for s in res[:1]:
        s = create_db(s)
        s = create_knn_match(s)
        s = copy_img(s)
        s = feature_extract(s)
        s = custom_match(s)
        s = tbaloop(s)

def __main():
    """run slow global colmap matcher"""
    from d3sim.algos.d3gs.hd3gs.data.load import _load_debug_scene
    scene = _load_debug_scene()
    scene = ColmapFeatureExtract(default_focal_length_factor=0.5)(scene)
    scene = ColmapCustomMatchGen()(scene)
    scene = ColmapCustomMatch()(scene)
    scene = ColmapMapper(is_hierarchical=True)(scene)
    # scene = ColmapUndistort()(scene)
    # scene = ColmapFetchReferenceResult(external_path="/root/autodl-tmp/example_dataset/camera_calibration/rectified/sparse/")(scene)

def __main_stage_2():
    """run fast postprocess"""
    from d3sim.algos.d3gs.hd3gs.data.load import _load_debug_scene
    scene = _load_debug_scene()
    scene = ColmapFeatureExtract(default_focal_length_factor=0.5)(scene)
    scene = ColmapCustomMatchGen()(scene)
    scene = ColmapCustomMatch()(scene)
    scene = ColmapMapper(is_hierarchical=True)(scene)
    # scene = ColmapUndistort()(scene)
    # scene = ColmapFetchReferenceResult(external_path="/root/autodl-tmp/example_dataset/camera_calibration/rectified/sparse/")(scene)

def __test_align_colmap_model():
    inp_model_path = "/root/autodl-tmp/example_dataset/camera_calibration/raw_chunks/0_0/bundle_adjustment/sparse/0/"
    old_model_path = "/root/autodl-tmp/example_dataset/camera_calibration/raw_chunks/0_0/sparse/0/"
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = tmpdir
        align_two_colmap_models(inp_model_path, old_model_path, out_path)
     

if __name__ == "__main__":
    __test_align_colmap_model()