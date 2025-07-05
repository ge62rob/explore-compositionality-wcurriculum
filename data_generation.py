import math
import random
import os
import csv

from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Any
from collections import namedtuple

import shapely
from shapely.geometry import Polygon, Point as ShapelyPoint, MultiPolygon
from shapely.ops import unary_union
from shapely.errors import GEOSException
import shapely.affinity

Point = namedtuple("Point", ["x", "y"])

# -------------------------------------------------------------------------
#                           Tangram Primitives
# -------------------------------------------------------------------------
SCALE_FACTOR = 40

large_triangle = [(0, 0), (2, 0), (0, 2)]
medium_triangle = [(0, 0), (math.sqrt(2), 0), (0, math.sqrt(2))]
small_triangle = [(0, 0), (1, 0), (0, 1)]
square = [(0, 0), (1, 0), (1, 1), (0, 1)]
parallelogram = [(0, 0), (1, 0), (1.5, 1), (0.5, 1)]

def scale_polygon(poly: List[Tuple[float, float]], factor: float) -> List[Point]:
    """Scales a polygon by a given factor."""
    return [Point(p[0] * factor, p[1] * factor) for p in poly]

LARGE_TRIANGLE = scale_polygon(large_triangle, SCALE_FACTOR)
MEDIUM_TRIANGLE = scale_polygon(medium_triangle, SCALE_FACTOR)
SMALL_TRIANGLE = scale_polygon(small_triangle, SCALE_FACTOR)
SQUARE = scale_polygon(square, SCALE_FACTOR)
PARALLELOGRAM = scale_polygon(parallelogram, SCALE_FACTOR)

TAN_TYPES = [
    ("large_triangle", LARGE_TRIANGLE),
    ("large_triangle", LARGE_TRIANGLE),
    ("medium_triangle", MEDIUM_TRIANGLE),
    ("small_triangle", SMALL_TRIANGLE),
    ("small_triangle", SMALL_TRIANGLE),
    ("square", SQUARE),
    ("parallelogram", PARALLELOGRAM),
]

ALLOWED_ROTATIONS = [i * math.pi / 4 for i in range(8)]  # 0°, 45°, ..., 315°
GRID_SIZE = 256
MAX_BOUND = 256

# -------------------------------------------------------------------------
#                           Basic Geometry Helpers
# -------------------------------------------------------------------------
def rotate_point(p: Point, angle: float) -> Point:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return Point(p.x * cos_a - p.y * sin_a, p.x * sin_a + p.y * cos_a)

def rotate_polygon(poly: List[Point], angle: float) -> List[Point]:
    return [rotate_point(p, angle) for p in poly]

def translate_polygon(poly: List[Point], dx: float, dy: float) -> List[Point]:
    return [Point(p.x + dx, p.y + dy) for p in poly]

def polygon_segments(poly: List[Point]) -> List[Tuple[Point, Point]]:
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]

def bounding_box(poly: List[Point]) -> Tuple[float, float, float, float]:
    xs = [p.x for p in poly]
    ys = [p.y for p in poly]
    return min(xs), max(xs), min(ys), max(ys)

def polygons_bounding_box(polys: List[List[Point]]) -> Tuple[float, float, float, float]:
    minx = min(p.x for poly in polys for p in poly)
    maxx = max(p.x for poly in polys for p in poly)
    miny = min(p.y for poly in polys for p in poly)
    maxy = max(p.y for poly in polys for p in poly)
    return minx, maxx, miny, maxy

def within_bounds(polys: List[List[Point]], limit: float = MAX_BOUND) -> bool:
    minx, maxx, miny, maxy = polygons_bounding_box(polys)
    if minx < 0 or miny < 0:
        return False
    if maxx > limit or maxy > limit:
        return False
    return True

def valid_polygon_or_none(coords) -> Polygon:
    """Create a Shapely Polygon, fix with buffer(0). Return None if invalid/empty."""
    try:
        poly = Polygon(coords)
        poly = poly.buffer(0)
        if poly.is_empty or not poly.is_valid or poly.area < 1e-12:
            return None
        return poly
    except GEOSException:
        return None

def polygons_intersect(poly1: List[Point], poly2: List[Point]) -> bool:
    coords1 = [(p.x, p.y) for p in poly1]
    coords2 = [(p.x, p.y) for p in poly2]

    p1 = valid_polygon_or_none(coords1)
    p2 = valid_polygon_or_none(coords2)
    if p1 is None or p2 is None:
        return False

    try:
        if not p1.intersects(p2):
            return False
        inter = p1.intersection(p2)
        return inter.area > 1e-9
    except GEOSException:
        return False

def any_intersection(new_poly: List[Point], placed: List[List[Point]]) -> bool:
    for p in placed:
        if polygons_intersect(new_poly, p):
            return True
    return False

def point_in_any_polygon(pt: Point, polys: List[List[Point]]) -> bool:
    sp = ShapelyPoint(pt.x, pt.y)
    for poly in polys:
        coords = [(p.x, p.y) for p in poly]
        shp = valid_polygon_or_none(coords)
        if shp is None:
            continue
        try:
            if shp.contains(sp):
                return True
        except GEOSException:
            continue
    return False

def piece_inside_another(new_poly: List[Point], placed: List[List[Point]]) -> bool:
    new_coords = [(pt.x, pt.y) for pt in new_poly]
    p = valid_polygon_or_none(new_coords)
    if p is None:
        return False

    # 1) Check if any vertex of new_poly is inside existing pieces
    for v in new_poly:
        if point_in_any_polygon(v, placed):
            return True

    # 2) Check centroid, bounding box midpoints, midpoints of edges
    cx, cy = p.centroid.x, p.centroid.y
    bx_min, bx_max, by_min, by_max = bounding_box(new_poly)

    candidates = [Point(cx, cy), Point((bx_min + bx_max) / 2, (by_min + by_max) / 2)]
    segs = polygon_segments(new_poly)
    for (a, b) in segs:
        candidates.append(Point((a.x + b.x) / 2, (a.y + b.y) / 2))

    for cpt in candidates:
        if point_in_any_polygon(cpt, placed):
            return True

    # 3) Check if any existing piece's vertex is inside new_poly
    for old_poly in placed:
        coords = [(pt.x, pt.y) for pt in old_poly]
        old_shp = valid_polygon_or_none(coords)
        if old_shp is None:
            continue
        for v in old_poly:
            sp = ShapelyPoint(v.x, v.y)
            try:
                if p.contains(sp):
                    return True
            except GEOSException:
                continue

    return False

def valid_placement(new_poly: List[Point], placed: List[List[Point]]) -> bool:
    if any_intersection(new_poly, placed):
        return False
    if piece_inside_another(new_poly, placed):
        return False
    if not within_bounds(placed + [new_poly]):
        return False
    return True

# -------------------------------------------------------------------------
#                Tangram Generation / Placement
# -------------------------------------------------------------------------
def get_vertices(poly: List[Point]) -> List[Point]:
    return poly

def edge_angle(p1: Point, p2: Point) -> float:
    return math.atan2(p2.y - p1.y, p2.x - p1.x)

def align_score_for_orientation(new_poly: List[Point], placed_poly: List[Point], attach_point: Point) -> int:
    """
    Encourages alignment of edges around 'attach_point'.
    """
    placed_vertices = get_vertices(placed_poly)
    try:
        idx_p = placed_vertices.index(attach_point)
    except ValueError:
        return 0

    prev_p = placed_vertices[idx_p - 1]
    next_p = placed_vertices[(idx_p + 1) % len(placed_vertices)]
    placed_angles = [edge_angle(attach_point, prev_p), edge_angle(attach_point, next_p)]

    new_vertices = get_vertices(new_poly)
    attach_new = new_vertices[0]
    prev_n = new_vertices[-1]
    next_n = new_vertices[1]
    new_angles = [edge_angle(attach_new, prev_n), edge_angle(attach_new, next_n)]

    score = 0
    for pa in placed_angles:
        for na in new_angles:
            diff = abs((pa - na) % (2 * math.pi))
            diff = min(diff, 2 * math.pi - diff)
            if diff < 1e-2:
                score += 1
    return score

def sample_orientation_distribution(
    base_poly: List[Point],
    new_poly: List[Point],
    placed: List[List[Point]],
    attach_point: Point,
    attach_vertex_new: Point
) -> List[Tuple[float, float]]:
    """
    We compute how well edges align to give preference to certain orientations.
    """
    dx = -attach_vertex_new.x
    dy = -attach_vertex_new.y
    new_centered = translate_polygon(new_poly, dx, dy)

    dx_b = -attach_point.x
    dy_b = -attach_point.y
    base_centered = translate_polygon(base_poly, dx_b, dy_b)

    weights = []
    for ang in ALLOWED_ROTATIONS:
        rotated = rotate_polygon(new_centered, ang)
        score = align_score_for_orientation(rotated, base_centered, Point(0, 0))
        w = 1 + score * 5.0
        weights.append((ang, w))

    total = sum(w for (_, w) in weights)
    if total < 1e-9:
        # uniform
        n = len(weights)
        return [(o, 1.0 / n) for (o, _) in weights]
    else:
        return [(o, w / total) for (o, w) in weights]

def try_place_tan(
    placed: List[List[Point]],
    new_tan: List[Point],
    max_attempts: int = 100
) -> Tuple[bool, List[Point]]:
    """Attempts random adjacency and orientation for new piece."""
    if not placed:
        return False, []

    placed_polys = placed[:]
    attempt_count = 0

    while attempt_count < max_attempts:
        attempt_count += 1
        base_poly = random.choice(placed_polys)
        base_vertices = get_vertices(base_poly)
        attach_point = random.choice(base_vertices)

        new_vertices = get_vertices(new_tan)
        attach_vertex_new = random.choice(new_vertices)

        orientation_dist = sample_orientation_distribution(
            base_poly,
            new_tan,
            placed_polys,
            attach_point,
            attach_vertex_new
        )
        tried_orientations = set()

        for _ in range(len(ALLOWED_ROTATIONS)):
            angles_weights = [(o, w) for (o, w) in orientation_dist if o not in tried_orientations]
            if not angles_weights:
                break
            angles, weights = zip(*angles_weights)
            chosen_ang = random.choices(angles, weights=weights)[0]

            # Translate new_tan so attach_vertex_new is at origin
            dx = -attach_vertex_new.x
            dy = -attach_vertex_new.y
            centered = translate_polygon(new_tan, dx, dy)

            # Rotate
            rotated = rotate_polygon(centered, chosen_ang)

            # Translate to attach_point
            final_placed = translate_polygon(rotated, attach_point.x, attach_point.y)

            if valid_placement(final_placed, placed_polys):
                return True, final_placed
            else:
                tried_orientations.add(chosen_ang)
                # set weight to 0, re-normalize
                orientation_dist = [(o, 0 if o == chosen_ang else w) for (o, w) in orientation_dist]
                sw = sum(w for (_, w) in orientation_dist)
                if sw < 1e-9:
                    break
                else:
                    orientation_dist = [(o, w / sw) for (o, w) in orientation_dist]

    return False, []

# -------------------------------------------------------------------------
# (A) Create "Solved" Version by Negative Buffer
# -------------------------------------------------------------------------
def create_solved_version(pieces: List[List[Point]], gap=1.0) -> List[List[Point]]:
    """
    Creates a "solved" version by shrinking each piece inward slightly
    so edges that were touching become visibly separated.

    If a piece is too small and collapses, we keep the original piece.
    """
    solved_pieces = []
    for poly in pieces:
        coords = [(p.x, p.y) for p in poly]
        shp = valid_polygon_or_none(coords)
        if shp is None:
            solved_pieces.append(poly)
            continue

        try:
            shrunk = shp.buffer(-gap)
            if shrunk.is_empty or shrunk.area < 1e-9:
                solved_pieces.append(poly)
            else:
                if isinstance(shrunk, MultiPolygon):
                    # pick the largest part if multipolygon
                    biggest_part = max(shrunk.geoms, key=lambda g: g.area)
                    shrunk = biggest_part
                shrunk_coords = list(shrunk.exterior.coords)
                new_poly = [Point(x, y) for (x, y) in shrunk_coords]
                solved_pieces.append(new_poly)
        except GEOSException:
            solved_pieces.append(poly)

    return solved_pieces

# -------------------------------------------------------------------------
# (B) Reflection Helpers for Symmetry Check
# -------------------------------------------------------------------------
def reflect_x(geom, cx):
    """Reflect across vertical line x=cx."""
    moved = shapely.affinity.translate(geom, xoff=-cx, yoff=0)
    flipped = shapely.affinity.scale(moved, xfact=-1, yfact=1, origin=(0, 0))
    result = shapely.affinity.translate(flipped, xoff=cx, yoff=0)
    return result

def reflect_y(geom, cy):
    """Reflect across horizontal line y=cy."""
    moved = shapely.affinity.translate(geom, xoff=0, yoff=-cy)
    flipped = shapely.affinity.scale(moved, xfact=1, yfact=-1, origin=(0, 0))
    result = shapely.affinity.translate(flipped, xoff=0, yoff=cy)
    return result

# -------------------------------------------------------------------------
# (C) Outline & Edge Measures
# -------------------------------------------------------------------------
def measure_tangram_outline(pieces: List[List[Point]]) -> Dict[str, Any]:
    """
    Computes "global" properties of the combined outline:
    - perimeter
    - num_holes
    - num_vertices (exterior)
    - x_range, y_range
    - convex_hull_pct
    - is_symmetric
    Also returns the union geometry to help get edges.
    """
    polygons = []
    for poly in pieces:
        coords = [(p.x, p.y) for p in poly]
        shp = valid_polygon_or_none(coords)
        if shp:
            polygons.append(shp)

    if not polygons:
        return {
            "perimeter": 0.0,
            "num_holes": 0,
            "num_vertices": 0,
            "x_range": 0.0,
            "y_range": 0.0,
            "convex_hull_pct": 0.0,
            "is_symmetric": False,
            "union_geom": None,
        }

    union_geom = unary_union(polygons)
    if union_geom.is_empty:
        return {
            "perimeter": 0.0,
            "num_holes": 0,
            "num_vertices": 0,
            "x_range": 0.0,
            "y_range": 0.0,
            "convex_hull_pct": 0.0,
            "is_symmetric": False,
            "union_geom": None,
        }

    if isinstance(union_geom, MultiPolygon):
        outline = unary_union(union_geom)
    else:
        outline = union_geom

    perimeter = outline.length
    polygons_list = []
    if isinstance(outline, MultiPolygon):
        polygons_list = list(outline.geoms)
    else:
        polygons_list = [outline]

    num_holes = 0
    for poly in polygons_list:
        num_holes += len(poly.interiors)

    if isinstance(outline, MultiPolygon):
        num_vertices = sum(len(poly.exterior.coords) for poly in outline.geoms)
    else:
        num_vertices = len(outline.exterior.coords)

    minx, miny, maxx, maxy = outline.bounds
    x_range = maxx - minx
    y_range = maxy - miny

    outline_area = outline.area
    hull_area = outline.convex_hull.area if not outline.convex_hull.is_empty else 1e-9
    convex_hull_pct = outline_area / hull_area if hull_area > 1e-9 else 1.0

    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0

    is_symmetric_x = False
    is_symmetric_y = False
    try:
        refl_x = reflect_x(outline, center_x)
        intersection_x = outline.intersection(refl_x)
        if intersection_x.area / outline_area > 0.95:
            is_symmetric_x = True
    except GEOSException:
        pass

    try:
        refl_y = reflect_y(outline, center_y)
        intersection_y = outline.intersection(refl_y)
        if intersection_y.area / outline_area > 0.95:
            is_symmetric_y = True
    except GEOSException:
        pass

    is_symmetric = (is_symmetric_x or is_symmetric_y)

    return {
        "perimeter": perimeter,
        "num_holes": num_holes,
        "num_vertices": num_vertices,
        "x_range": x_range,
        "y_range": y_range,
        "convex_hull_pct": convex_hull_pct,
        "is_symmetric": is_symmetric,
        "union_geom": outline,
    }

def get_longest_and_shortest_edges(union_geom) -> Tuple[float, float]:
    """
    Finds the longest and shortest edges in the union outline (exterior).
    If the union is MultiPolygon, we check all exteriors.
    """
    if (not union_geom) or union_geom.is_empty:
        return (0.0, 0.0)

    if isinstance(union_geom, MultiPolygon):
        polys = list(union_geom.geoms)
    else:
        polys = [union_geom]

    edges = []
    for p in polys:
        coords = list(p.exterior.coords)
        for i in range(len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % len(coords)]
            dist = math.hypot(x2 - x1, y2 - y1)
            edges.append(dist)

    if not edges:
        return (0.0, 0.0)

    longest = max(edges)
    shortest = min(edges)
    return (longest, shortest)

# -------------------------------------------------------------------------
# (D) Additional Helpers: number of matched points
# -------------------------------------------------------------------------
def count_matched_points(pieces: List[List[Point]]) -> int:
    """
    The number of pairs of points that lie at the same position.
    For each (x,y), how many times does it appear among all piece vertices?
    If it appears N times, that yields (N-1) "repeats".
    We'll do sum((N-1)) for all coords.
    """
    from collections import defaultdict
    freq = defaultdict(int)

    for poly in pieces:
        for pt in poly:
            freq[(round(pt.x, 4), round(pt.y, 4))] += 1

    matched_points_count = 0
    for coord, count in freq.items():
        if count > 1:
            matched_points_count += (count - 1)
    return matched_points_count

# -------------------------------------------------------------------------
# (E) Paper-Derived Difficulty Metric
# -------------------------------------------------------------------------
def measure_paper_difficulty(pieces: List[List[Point]]) -> float:
    """
    A single numeric "difficulty" measure that combines:
      - perimeter
      - number of holes
      - convex hull percentage
      - longest edge / shortest edge ratio
      - number of matched points
      - symmetry => a slight decrease in difficulty if symmetrical
    """
    outline_info = measure_tangram_outline(pieces)
    union_geom = outline_info["union_geom"]
    if (not union_geom) or union_geom.is_empty:
        return 0.0

    perimeter = outline_info["perimeter"]
    num_holes = outline_info["num_holes"]
    convex_pct = outline_info["convex_hull_pct"]
    is_symmetric = outline_info["is_symmetric"]

    # longest / shortest
    longest, shortest = get_longest_and_shortest_edges(union_geom)
    edge_ratio = 0.0 if shortest < 1e-9 else (longest / shortest)

    matched_points_count = count_matched_points(pieces)

    # Weighted sum
    difficulty = 0.0
    difficulty += 0.02 * perimeter
    difficulty += 5.0 * num_holes
    difficulty += 8.0 * convex_pct
    difficulty += edge_ratio
    difficulty += 0.5 * matched_points_count

    if is_symmetric:
        difficulty *= 0.9  # reduce by 10%

    return difficulty

# -------------------------------------------------------------------------
# (F) Drawing & Saving
# -------------------------------------------------------------------------
def draw_tangram_pil(pieces: List[List[Point]], img_size: int = 256) -> Image.Image:
    """Draw tangram pieces in white on black background."""
    img = Image.new("RGB", (img_size, img_size), "black")
    draw = ImageDraw.Draw(img)
    for piece in pieces:
        # Because y-axis in PIL is downward, invert piece.y => (x, img_size - y)
        xy = [(p.x, img_size - p.y) for p in piece]
        draw.polygon(xy, fill="white")
    return img

def save_tangram_images():
    """
    Generates tangram images for 1..7 pieces, 100 samples each by default.
    *Images (NORMAL)* -> 'dataset/tangrams_{n}_piece/'
    *Images (SOLVED)* -> 'dataset_solved/tangrams_{n}_piece/'
    *All CSV metrics into a SINGLE file* located in 'dataset_csv/tangram_metrics_all.csv'.
    """
    # How many samples per piece-count?
    samples_per_piece = 1000

    # 1) Create single CSV in 'dataset_csv'
    csv_folder = "dataset_csv"
    os.makedirs(csv_folder, exist_ok=True)
    csv_all_path = os.path.join(csv_folder, "tangram_metrics_all.csv")

    fieldnames = [
        "num_pieces",
        "filename_normal",
        "filename_solved",
        "num_vertices",
        "num_holes",
        "perimeter",
        "x_range",
        "y_range",
        "convex_hull_pct",
        "is_symmetric",
        "longest_edge",
        "shortest_edge",
        "matched_points",
        "paper_difficulty",
    ]

    with open(csv_all_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # For each piece-count 1..7
        for num_pieces in range(1, 8):
            # Make subfolders
            normal_dir = f"dataset/tangrams_{num_pieces}_piece"
            solved_dir = f"dataset_solved/tangrams_{num_pieces}_piece"
            os.makedirs(normal_dir, exist_ok=True)
            os.makedirs(solved_dir, exist_ok=True)

            print(f"Generating {samples_per_piece} images for {num_pieces} piece(s)...")

            for idx in range(samples_per_piece):
                placed_polys = []
                success = True
                
                if num_pieces == 1:
                    # For 1-piece tangrams, first create a 7-piece tangram
                    # and then randomly remove pieces until only one remains
                    shapes_selected = random.sample(TAN_TYPES, 7)
                    random.shuffle(shapes_selected)
                    
                    # Place first piece in the center
                    sname, spoly = shapes_selected[0]
                    first_angle = random.choice(ALLOWED_ROTATIONS)
                    rotated_first = rotate_polygon(spoly, first_angle)
                    grid_center = GRID_SIZE // 2
                    placed_polys.append(translate_polygon(rotated_first, grid_center, grid_center))
                    
                    # Place remaining pieces
                    for (nm, poly) in shapes_selected[1:]:
                        ok, newp = try_place_tan(placed_polys, poly, max_attempts=500)
                        if not ok:
                            success = False
                            break
                        placed_polys.append(newp)
                    
                    # If successful, randomly remove pieces until only one remains
                    if success and within_bounds(placed_polys):
                        # Keep removing pieces randomly until only one remains
                        while len(placed_polys) > 1:
                            # Randomly select a piece to keep
                            piece_to_remove = random.randrange(len(placed_polys))
                            placed_polys.pop(piece_to_remove)
                    else:
                        # If 7-piece generation failed, fall back to standard 1-piece generation
                        shapes_selected = random.sample(TAN_TYPES, 1)
                        sname, spoly = shapes_selected[0]
                        first_angle = random.choice(ALLOWED_ROTATIONS)
                        rotated_first = rotate_polygon(spoly, first_angle)
                        grid_center = GRID_SIZE // 2
                        placed_polys = [translate_polygon(rotated_first, grid_center, grid_center)]
                else:
                    # Standard method for 2-7 pieces
                    shapes_selected = random.sample(TAN_TYPES, num_pieces)
                    random.shuffle(shapes_selected)

                    # Place first piece in the center
                    sname, spoly = shapes_selected[0]
                    first_angle = random.choice(ALLOWED_ROTATIONS)
                    rotated_first = rotate_polygon(spoly, first_angle)
                    grid_center = GRID_SIZE // 2
                    placed_polys.append(translate_polygon(rotated_first, grid_center, grid_center))

                    # Place remaining
                    for (nm, poly) in shapes_selected[1:]:
                        ok, newp = try_place_tan(placed_polys, poly, max_attempts=500)
                        if not ok:
                            success = False
                            break
                        placed_polys.append(newp)

                # If failed or out of bounds
                normal_filename = f"{num_pieces}piece_{idx + 1:04d}_normal.png"
                solved_filename = f"{num_pieces}piece_{idx + 1:04d}_solved.png"
                normal_path = os.path.join(normal_dir, normal_filename)
                solved_path = os.path.join(solved_dir, solved_filename)

                if (not success) or (not within_bounds(placed_polys)):
                    # black images
                    normal_img = Image.new("RGB", (256, 256), "black")
                    solved_img = Image.new("RGB", (256, 256), "black")
                    normal_img.save(normal_path)
                    solved_img.save(solved_path)

                    writer.writerow({
                        "num_pieces": num_pieces,
                        "filename_normal": normal_filename,
                        "filename_solved": solved_filename,
                        "num_vertices": 0,
                        "num_holes": 0,
                        "perimeter": 0,
                        "x_range": 0,
                        "y_range": 0,
                        "convex_hull_pct": 0,
                        "is_symmetric": False,
                        "longest_edge": 0,
                        "shortest_edge": 0,
                        "matched_points": 0,
                        "paper_difficulty": 999.0,
                    })
                    print(f"Warning: Failed generation for {num_pieces} pieces at sample {idx+1}.")
                    continue

                # Otherwise draw normal tangram
                tangram = placed_polys
                normal_img = draw_tangram_pil(tangram, img_size=256)
                normal_img.save(normal_path)

                # Make "solved" version -> separate folder
                solved_pieces = create_solved_version(tangram, gap=1.0)
                solved_img = draw_tangram_pil(solved_pieces, img_size=256)
                solved_img.save(solved_path)

                # Measure
                outline_info = measure_tangram_outline(tangram)
                perimeter = outline_info["perimeter"]
                num_holes = outline_info["num_holes"]
                num_vertices = outline_info["num_vertices"]
                x_range = outline_info["x_range"]
                y_range = outline_info["y_range"]
                convex_hull_pct = outline_info["convex_hull_pct"]
                is_symmetric = outline_info["is_symmetric"]
                union_geom = outline_info["union_geom"]

                longest_edge, shortest_edge = get_longest_and_shortest_edges(union_geom)
                matched_points = count_matched_points(tangram)
                paper_diff = measure_paper_difficulty(tangram)

                record = {
                    "num_pieces": num_pieces,
                    "filename_normal": normal_filename,
                    "filename_solved": solved_filename,
                    "num_vertices": num_vertices,
                    "num_holes": num_holes,
                    "perimeter": round(perimeter, 3),
                    "x_range": round(x_range, 3),
                    "y_range": round(y_range, 3),
                    "convex_hull_pct": round(convex_hull_pct, 3),
                    "is_symmetric": is_symmetric,
                    "longest_edge": round(longest_edge, 3),
                    "shortest_edge": round(shortest_edge, 3),
                    "matched_points": matched_points,
                    "paper_difficulty": round(paper_diff, 3),
                }
                writer.writerow(record)
                print(f"[{normal_filename}] diff={paper_diff:.3f}, perimeter={perimeter:.1f}, holes={num_holes}")

            print(f"Completed {samples_per_piece} images for {num_pieces} pieces.\n")

    print("All tangram images have been generated and saved.")
    print(f"Metrics CSV file located at: {csv_all_path}")


if __name__ == "__main__":
    save_tangram_images()
