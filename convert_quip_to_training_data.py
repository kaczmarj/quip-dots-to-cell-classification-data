"""Create image tiles centered on each dot annotation inside a labeled nucleus.

This script was written for PAAD nuclei classification. Nuclei polygons (without
associated class labels) were taken from Le Hou's Scientific Data paper. A pathologist
dot annotated some of these nuclei. This script takes each dot annotation, finds the
corresponding nucleus in Le Hou's data, and saves an image of the whole slide image
with the polygon at the center of the image.

Example
-------
python convert_quip_to_training_data.py \\
    --dump-path /data/quip_distro/data/dump/TCGA:paad-2021-7-30-17-47-57/ \\
    --svs-root /data/quip_distro/images/tcga_data/paad/ \\
    --polygon-root /data/quip_distro/data/kaczmarj/hou-scientific-data-nuclei-v0/paad_polygon/ \\
    --roi-name "Waqas 500p Non-Tumor 500p" \\
    --label-regex Waqas \\
    --subject-ids TCGA-2J-AAB9 TCGA-2J-AABA \\
    --save-polygon-overlays \\
    --output-dir outputs

Before running this script, human annotations need to be dumped from QuIP. The username and
password are your credentials for the QuIP web viewer, not the quip server.

/data/quip_distro/data/bin/dumphuman \\
    -username <quip username> \\
    -password <quip password> \\
    -collectionname <collection name>

This will create a dumped directory (see path in preceding example).
"""

import argparse
from collections import Counter
from collections import namedtuple
import json
from pathlib import Path
import re
import typing as ty

import openslide
import pandas as pd
from PIL import ImageDraw
from shapely import affinity
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import box as _box
from shapely.ops import unary_union as _unary_union

# Custom types.
PathType = ty.Union[str, Path]
TileBoundingBox = namedtuple("bbox", "x0 y0 x1 y1 path")


def _get_bboxes_from_filenames(paths: ty.Sequence[Path]) -> ty.List[TileBoundingBox]:
    """Convert filenames to bounding box objects."""
    bboxes = []
    for p in paths:
        psplit = p.name.split("_")
        x0, y0 = int(psplit[0]), int(psplit[1])
        x1 = x0 + int(psplit[2])
        y1 = y0 + int(psplit[3])
        bboxes.append(TileBoundingBox(x0, y0, x1, y1, p))
    return bboxes


def _point_in_tilebbox(bbox: TileBoundingBox, point: ty.Tuple[float, float]) -> bool:
    """Return whether an XY point is inside a tile bounding box."""
    in_x = bbox.x0 <= point[0] <= bbox.x1
    in_y = bbox.y0 <= point[1] <= bbox.y1
    return in_x and in_y


def _get_human_rois_as_multi_polygon(
    json_data: ty.List[ty.Dict[str, ty.Any]],
    roi_name: str,
    slide_width: int,
    slide_height: int,
    bbox_size: int = 500,
) -> MultiPolygon:
    """Return a shapely MultiPolygon representing all of the human ROIs.

    The human ROIs are the boxes that are placed on a slide and in which nuclei are
    dotted.
    """

    def get_polygon_from_label(label):
        coords = label["geometries"]["features"][0]["bound"]["coordinates"][0]
        coords = [(int(x * slide_width), int(y * slide_height)) for x, y in coords]
        minx = min(coords, key=lambda x: x[0])[0]
        miny = min(coords, key=lambda x: x[1])[1]
        maxx = max(coords, key=lambda x: x[0])[0] + bbox_size
        maxy = max(coords, key=lambda x: x[1])[1] + bbox_size
        return _box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    rois = [d for d in json_data if d["properties"]["annotations"]["notes"] == roi_name]
    bboxes = [get_polygon_from_label(roi) for roi in rois]
    bboxes_union = _unary_union(bboxes)
    return bboxes_union


def _chunks(lst: ty.Sequence, n: int) -> ty.Generator[ty.Sequence, None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]  # noqa: E203


def _get_polygons_containing_point(
    features_csv: PathType,
    point: Point,
) -> ty.List[Polygon]:
    """Return list of shapely Polygon instances that contain a point."""
    df = pd.read_csv(features_csv)
    all_pp = df.loc[:, "Polygon"].values.tolist()
    # Convert quip format of polygon points to a nested list
    # [[(x,y), (x1,y1), ...], [(x5, y5), ...], ...]
    all_pp = (_chunks([float(x) for x in pp[1:-1].split(":")], 2) for pp in all_pp)
    all_pp = (Polygon(points) for points in all_pp)
    return list(filter(lambda polygon: polygon.contains(point), all_pp))


def _get_upper_left(
    centroid: ty.Tuple[float, float],
    size: ty.Tuple[int, int] = (256, 256),
) -> ty.Tuple[int, int]:
    """Given an XY point, return the upper-left point so that XY is at the center."""
    x = centroid[0] - size[0] / 2
    y = centroid[1] - size[1] / 2
    return round(x), round(y)


def _make_tile_filename(
    slide_path: Path,
    upper_left: ty.Tuple[int, int],
) -> str:
    """Create unique tile filename."""
    return f"{slide_path.name}_{upper_left[0]}_{upper_left[1]}.png"


# Workflow...
# For each JSON file in the dump, get each label (i.e., point).
# For each JSON file, get the path to the corresponding whole slide image.
# If there are human-made ROIs, check for each point if the point is inside the ROIs.
# For each label in the JSON file, find the file in Le Hou's data that contains the
# point. Then look through that file and find the polygon that contains the point.
# Check that there is only one.
# Get the region of the whole slide image with the polygon at the center, using the
# centroid of the polygon.
# Save the image as a PNG with associated class information. Can also save some
# metadata about the nucleus in the PNG file but not necessary. Better to save the
# actual polygon coordinates.
def save_tiles(
    output_size: ty.Tuple[int, int],
    output_dir: PathType,
    svs_root: PathType,
    polygon_root: PathType,
    dump_path: PathType,
    roi_name: str = None,
    label_regex: str = None,
    subject_ids: ty.Iterable[str] = None,
    save_polygon_overlays: bool = False,
):
    """Save regions of whole slide image around polygons to disk.

    Parameters
    ----------
    output_size : (int, int)
        Size of output images in pixels.
    output_dir : str or Path
        Directory in which to save outputs.
    svs_root : str or Path
        Path to the whole slide image SVS files.
    polygon_root : str or Path
        Path to the polygons (e.g., *-features.csv) from Hou et al. (Scientific Data).
    dump_path : str or Path
        Path to the data dump on QuIP.
    roi_name : str, optional
        The name of the label that defines the region of interest, inside which nuclei
        are annotated. If provided, only dots contained inside the polygons named
        <roi_name> are included.
    label_regex : str, optional
        Regular expression for the label names. If provided, will only save labels that
        match the expression.
    subject_ids : list of str, optional
        Subject IDs to take labels from. By default, take labels from all slides.
    save_polygon_overlays : bool, optional
        Save copies of the tiles with the polygon overlaid. These images are saved in
        directories with prefix "poly-".
    """
    output_dir = Path(output_dir)
    svs_root = Path(svs_root)
    polygon_root = Path(polygon_root)
    dump_path = Path(dump_path)

    df_manifest = pd.read_csv(dump_path / "manifest.csv")
    if subject_ids:
        _mask = df_manifest.loc[:, "clinicaltrialsubjectid"].isin(subject_ids)
        df_manifest = df_manifest.loc[_mask, :]
        del _mask
    print(f"Getting data from {df_manifest.shape[0]} images...")
    # dict of JSON_PATH: ORIGINAL_SVS_PATH
    json_path_to_svs = dict(zip(df_manifest.path, df_manifest.imagepath))

    if label_regex is not None:
        label_regex = re.compile(label_regex)

    for json_file, original_svs_path in json_path_to_svs.items():
        original_svs_path = Path(original_svs_path)
        print(f"Working on {original_svs_path.name} ...")
        print(f"Reading {dump_path / json_file} ...")
        with open(dump_path / json_file) as f:
            json_data = json.load(f)
        print(f"Found {len(json_data):,} labels")

        oslide = openslide.OpenSlide(str(svs_root / original_svs_path.name))

        # If the name of the human ROI label is given, we create a polygon representing
        # the union of all of those regions. We keep dots that are inside those regions.
        human_rois: ty.Optional[MultiPolygon] = None
        if roi_name is not None:
            human_rois = _get_human_rois_as_multi_polygon(
                json_data=json_data,
                roi_name=roi_name,
                slide_width=oslide.dimensions[0],
                slide_height=oslide.dimensions[1],
                bbox_size=500,  # TODO: we shouldn't hardcode this but for now it's ok.
            )

        # Keep only "point" types.
        json_data = [
            d
            for d in json_data
            if d["geometries"]["features"][0]["geometry"]["type"] == "Point"
        ]
        print(f"{len(json_data):,} labels are points (only keeping these)")

        if any(len(d["geometries"]["features"]) != 1 for d in json_data):
            print("WARNING: at least one label has more than one point")
            print("WARNING: will take the first point for those cases")

        print("Number of points per class:")
        _tmp = Counter(d["properties"]["annotations"]["notes"] for d in json_data)
        for k, v in _tmp.items():
            print(f"    {k}: {v}")
        del _tmp

        _to_glob_polygons = polygon_root / original_svs_path.name
        print(f"Searching for polygons in {_to_glob_polygons}")
        if not _to_glob_polygons.exists():
            print("SKIPPING because directory does not exist")
            continue
        polygon_paths = list(_to_glob_polygons.glob("*.csv"))
        del _to_glob_polygons
        if not polygon_paths:
            print("SKIPPING because did not find any polygon CSV files")
            continue
        print(f"Found {len(polygon_paths):,} CSV files with polygons")

        bboxes = _get_bboxes_from_filenames(polygon_paths)

        for human_label in json_data:
            class_label: str = human_label["properties"]["annotations"]["notes"]
            if label_regex is not None:
                if label_regex.match(class_label) is None:
                    print("SKIPPING label because does not match label regex")
                    continue

            point: ty.Tuple[float, float] = human_label["geometries"]["features"][0][
                "geometry"
            ]["coordinates"]
            if len(point) != 2:
                print("SKIPPING label because this point does not have two values")
                continue

            point[0] *= oslide.dimensions[0]
            point[1] *= oslide.dimensions[1]

            if human_rois is not None:
                if not human_rois.contains(Point(*point)):
                    print("SKIPPING label because outside of human ROIs")
                    continue

            bboxes_containing_point = [
                b for b in bboxes if _point_in_tilebbox(b, point)
            ]
            if not bboxes_containing_point:
                print("SKIPPING label because could not find corresponding feature CSV")
                continue
            elif len(bboxes_containing_point) > 1:
                raise ValueError("found more than one matched feature CSV... what do?")

            polygons_containing_point = _get_polygons_containing_point(
                features_csv=bboxes_containing_point[0].path,
                point=Point(*point),
            )
            if not polygons_containing_point:
                print("SKIPPING label because could not find corresponding polygon")
                continue
            elif len(polygons_containing_point) > 1:
                print("SKIPPING because label matches more than one polygon. what do?")
                continue

            polygon_containing_point = polygons_containing_point[0]
            del polygons_containing_point

            x, y = polygon_containing_point.centroid.coords.xy
            centroid = x[0], y[0]
            del x, y

            upper_left = _get_upper_left(centroid, size=output_size)

            output_name = _make_tile_filename(
                slide_path=svs_root / original_svs_path.name,
                upper_left=upper_left,
            )

            output_path = output_dir / class_label.replace(" ", "_") / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            img = oslide.read_region(upper_left, level=0, size=output_size)
            print(f"SAVING image to {output_path}")
            img.save(output_path)

            # TODO: save images with overlaid nucleus polygon
            if save_polygon_overlays:
                _tmp_polygon = affinity.translate(
                    polygon_containing_point,
                    xoff=-upper_left[0],
                    yoff=-upper_left[1],
                )
                draw = ImageDraw.Draw(img)
                _tmp_x, _tmp_y = _tmp_polygon.exterior.xy
                draw.polygon(list(zip(_tmp_x, _tmp_y)), outline="green")
                _tmp_output_path = (
                    output_dir / ("poly-" + class_label.replace(" ", "_")) / output_name
                )
                _tmp_output_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"SAVING image with polygon overlay to {_tmp_output_path}")
                img.save(_tmp_output_path)
                del _tmp_polygon, _tmp_x, _tmp_y, _tmp_output_path


def _get_parsed_args(args=None) -> argparse.Namespace:
    """Return namespace of parsed command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dump-path", required=True, type=Path, help="Path to data dump.")
    p.add_argument("--svs-root", required=True, type=Path, help="Path to SVS files.")
    p.add_argument(
        "--polygon-root",
        type=Path,
        required=True,
        help="Path to *-features.csv files containing nuclei segmentations.",
    )
    p.add_argument(
        "--output-size",
        default=(256, 256),
        nargs=2,
        type=int,
        help="Size of output images (width, height).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Path in which to save images.",
    )
    p.add_argument(
        "--label-regex",
        default=None,
        help="Only keep regular expressions that match this label.",
    )
    p.add_argument(
        "--roi-name",
        default=None,
        help="Name of human bounding box label. Assumed to be 500px in size.",
    )
    p.add_argument(
        "--subject-ids",
        nargs="+",
        help="Only use images from these subject IDs. By default, use all images.",
    )
    p.add_argument(
        "--save-polygon-overlays",
        action="store_true",
        help="Save a copy of each tile with the polygon overlaid on the nucleus.",
    )
    return p.parse_args(args)


if __name__ == "__main__":
    args = _get_parsed_args()
    save_tiles(
        output_size=args.output_size,
        output_dir=args.output_dir,
        svs_root=args.svs_root,
        polygon_root=args.polygon_root,
        dump_path=args.dump_path,
        roi_name=args.roi_name,
        label_regex=args.label_regex,
        subject_ids=args.subject_ids,
        save_polygon_overlays=args.save_polygon_overlays,
    )
