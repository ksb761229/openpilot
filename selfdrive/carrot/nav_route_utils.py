import math
from typing import List, Tuple, Optional

from shapely.geometry import LineString


PointLL = Tuple[float, float]   # (lon, lat)
PointXY = Tuple[float, float]   # (x, y) meters


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
  R = 6371000.0
  phi1 = math.radians(lat1)
  phi2 = math.radians(lat2)
  dphi = math.radians(lat2 - lat1)
  dlambda = math.radians(lon2 - lon1)

  a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
  return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))


def ll_to_local_xy(lon: float, lat: float, ref_lon: float, ref_lat: float) -> PointXY:
  x = (lon - ref_lon) * 40008000.0 * math.cos(math.radians(ref_lat)) / 360.0
  y = (lat - ref_lat) * 40008000.0 / 360.0
  return x, y


def rotate_to_vehicle_frame(x: float, y: float, heading_deg: float) -> PointXY:
  """
  world-local xy -> vehicle frame
  return: (forward, left)
  """
  h = math.radians(heading_deg)
  forward = x * math.cos(h) + y * math.sin(h)
  left = -x * math.sin(h) + y * math.cos(h)
  return forward, left


def project_point_to_segment_xy(p1: PointXY, p2: PointXY, p: PointXY) -> Tuple[PointXY, float]:
  x1, y1 = p1
  x2, y2 = p2
  px, py = p

  dx = x2 - x1
  dy = y2 - y1
  denom = dx * dx + dy * dy
  if denom <= 1e-9:
    return p1, 0.0

  t = ((px - x1) * dx + (py - y1) * dy) / denom
  t = max(0.0, min(1.0, t))
  return (x1 + t * dx, y1 + t * dy), t


def find_closest_point_on_path(
  coordinates: List[PointLL],
  current_position: PointLL,
  start_index: int = 0,
  search_back: int = 3,
  search_ahead: int = 50,
) -> Tuple[int, Optional[PointLL], float]:
  """
  Returns:
    closest_segment_index,
    closest_point(lon, lat),
    min_distance_m
  """
  if len(coordinates) < 2:
    return -1, None, float("inf")

  ref_lon, ref_lat = current_position
  cur_xy = (0.0, 0.0)

  begin = max(0, start_index - search_back)
  end = min(len(coordinates) - 1, max(begin + 1, start_index + search_ahead))

  min_dist = float("inf")
  closest_idx = -1
  closest_ll = None

  for i in range(begin, end):
    p1_ll = coordinates[i]
    p2_ll = coordinates[i + 1]

    p1_xy = ll_to_local_xy(p1_ll[0], p1_ll[1], ref_lon, ref_lat)
    p2_xy = ll_to_local_xy(p2_ll[0], p2_ll[1], ref_lon, ref_lat)

    proj_xy, t = project_point_to_segment_xy(p1_xy, p2_xy, cur_xy)
    dist = math.hypot(proj_xy[0], proj_xy[1])

    if dist < min_dist:
      min_dist = dist
      closest_idx = i
      closest_ll = (
        p1_ll[0] + t * (p2_ll[0] - p1_ll[0]),
        p1_ll[1] + t * (p2_ll[1] - p1_ll[1]),
      )

  return closest_idx, closest_ll, min_dist


def extract_forward_path(
  coordinates: List[PointLL],
  current_position: PointLL,
  start_index: int,
  lookahead_m: float,
) -> Tuple[List[PointLL], int, Optional[PointLL]]:
  if len(coordinates) < 2:
    return [], start_index, None

  closest_index, closest_point, _ = find_closest_point_on_path(
    coordinates,
    current_position,
    start_index=start_index,
  )

  if closest_index < 0 or closest_point is None:
    return [], start_index, None

  path = [closest_point]
  total_distance = 0.0

  next_pt = coordinates[closest_index + 1]
  first_seg = haversine_m(closest_point[0], closest_point[1], next_pt[0], next_pt[1])

  if first_seg > 0.01:
    path.append(next_pt)
    total_distance += first_seg

  for i in range(closest_index + 1, len(coordinates) - 1):
    p1 = coordinates[i]
    p2 = coordinates[i + 1]
    seg_dist = haversine_m(p1[0], p1[1], p2[0], p2[1])

    if seg_dist <= 0.01:
      continue

    if total_distance + seg_dist >= lookahead_m:
      remain = lookahead_m - total_distance
      ratio = remain / seg_dist
      interp_pt = (
        p1[0] + ratio * (p2[0] - p1[0]),
        p1[1] + ratio * (p2[1] - p1[1]),
      )
      path.append(interp_pt)
      break

    total_distance += seg_dist
    path.append(p2)

  return path, closest_index, closest_point


def gps_path_to_vehicle_xy(
  gps_path: List[PointLL],
  reference_point: PointLL,
  heading_deg: float,
) -> List[PointXY]:
  ref_lon, ref_lat = reference_point
  out: List[PointXY] = []

  for lon, lat in gps_path:
    x, y = ll_to_local_xy(lon, lat, ref_lon, ref_lat)
    forward, left = rotate_to_vehicle_frame(x, y, heading_deg)
    out.append((forward, left))

  return out


def resample_polyline(points_xy: List[PointXY], ds: float) -> Tuple[List[PointXY], List[float]]:
  if len(points_xy) < 2:
    return points_xy[:], [0.0] if points_xy else []

  line = LineString(points_xy)
  if line.length <= 1e-6:
    return [points_xy[0]], [0.0]

  sampled = []
  sampled_distances = []
  s = 0.0
  while s <= line.length:
    p = line.interpolate(s)
    sampled.append((p.x, p.y))
    sampled_distances.append(s)
    s += ds

  if sampled_distances[-1] < line.length:
    p = line.interpolate(line.length)
    sampled.append((p.x, p.y))
    sampled_distances.append(line.length)

  return sampled, sampled_distances


def smooth_polyline(points_xy: List[PointXY], window_size: int = 5) -> List[PointXY]:
  if len(points_xy) < 3 or window_size <= 1:
    return points_xy[:]

  w = max(3, window_size)
  if w % 2 == 0:
    w += 1
  half = w // 2

  out: List[PointXY] = []
  n = len(points_xy)

  for i in range(n):
    s = max(0, i - half)
    e = min(n, i + half + 1)
    xs = [points_xy[j][0] for j in range(s, e)]
    ys = [points_xy[j][1] for j in range(s, e)]
    out.append((sum(xs) / len(xs), sum(ys) / len(ys)))

  return out


def wrap_to_pi(a: float) -> float:
  while a > math.pi:
    a -= 2.0 * math.pi
  while a < -math.pi:
    a += 2.0 * math.pi
  return a


def compute_headings(points_xy: List[PointXY]) -> List[float]:
  hs = []
  for i in range(len(points_xy) - 1):
    dx = points_xy[i + 1][0] - points_xy[i][0]
    dy = points_xy[i + 1][1] - points_xy[i][1]
    hs.append(math.atan2(dy, dx))
  return hs


def compute_multiscale_curvature(
  points_xy: List[PointXY],
  ds: float,
  spans=(2, 4, 8),   # ds=3m -> 6m / 12m / 24m
) -> List[float]:
  hs = compute_headings(points_xy)
  n = len(points_xy)
  curv = [0.0] * n

  for i in range(n):
    vals = []
    for span in spans:
      j0 = i - span
      j1 = i + span
      if j0 < 0 or j1 >= len(hs):
        continue

      dtheta = wrap_to_pi(hs[j1] - hs[j0])
      ds_total = (j1 - j0) * ds
      if ds_total > 1e-6:
        vals.append(abs(dtheta) / ds_total)

    if vals:
      vals = sorted(vals)
      curv[i] = vals[len(vals) // 2]   # median
    else:
      curv[i] = 0.0

  return curv


def compute_local_turn_angle(
  points_xy: List[PointXY],
  ds: float,
  radius_m: float = 18.0,
) -> List[float]:
  hs = compute_headings(points_xy)
  span = max(1, int(radius_m / ds))
  n = len(points_xy)
  out = [0.0] * n

  if len(hs) < 2:
    return out

  for i in range(n):
    j0 = max(0, i - span)
    j1 = min(len(hs) - 1, i + span)
    if j1 <= j0:
      continue
    out[i] = abs(math.degrees(wrap_to_pi(hs[j1] - hs[j0])))

  return out


def curvature_to_speed_kph(
  curvatures: List[float],
  turn_angles_deg: List[float],
  vmax_kph: float = 110.0,
  a_lat_curve: float = 1.8,
  turn_angle_soft: float = 35.0,
  turn_angle_mid: float = 50.0,
  turn_angle_hard: float = 70.0,
  turn_speed_soft_kph: float = 35.0,
  turn_speed_mid_kph: float = 24.0,
  turn_speed_hard_kph: float = 18.0,
) -> List[float]:
  speeds = []

  for k, ang in zip(curvatures, turn_angles_deg):
    k_eff = max(k, 1e-4)
    v_curve = math.sqrt(a_lat_curve / k_eff) * 3.6
    v_curve = min(v_curve, vmax_kph)

    if ang >= turn_angle_hard:
      v_curve = min(v_curve, turn_speed_hard_kph)
    elif ang >= turn_angle_mid:
      v_curve = min(v_curve, turn_speed_mid_kph)
    elif ang >= turn_angle_soft:
      v_curve = min(v_curve, turn_speed_soft_kph)

    speeds.append(v_curve)

  return speeds


def apply_backward_speed_limit(
  speeds_kph: List[float],
  ds: float,
  decel_mps2: float = 1.8,
) -> List[float]:
  if not speeds_kph:
    return []

  out = [0.0] * len(speeds_kph)
  out[-1] = speeds_kph[-1] / 3.6
  a = max(0.1, decel_mps2)

  for i in range(len(speeds_kph) - 2, -1, -1):
    v_next = out[i + 1]
    v_allowed = math.sqrt(max(0.0, v_next * v_next + 2.0 * a * ds))
    out[i] = min(speeds_kph[i] / 3.6, v_allowed)

  return [v * 3.6 for v in out]


def build_navi_speed_profile(
  coordinates: List[PointLL],
  current_position: PointLL,
  heading_deg: float,
  start_index: int,
  lookahead_m: float = 200.0,
  ds: float = 3.0,
  smooth_window: int = 5,
  vmax_kph: float = 110.0,
  decel_mps2: float = 1.8,
):
  """
  Returns:
    display_points_xy,
    display_distances,
    current_speed_kph,
    next_start_index,
    closest_point,
    final_speeds_kph
  """
  path_gps, next_start_index, closest_point = extract_forward_path(
    coordinates=coordinates,
    current_position=current_position,
    start_index=start_index,
    lookahead_m=lookahead_m,
  )

  if not path_gps or closest_point is None:
    return [], [], 300.0, start_index, None, []

  path_xy = gps_path_to_vehicle_xy(
    gps_path=path_gps,
    reference_point=closest_point,
    heading_deg=heading_deg,
  )

  sampled_xy, sampled_distances = resample_polyline(path_xy, ds=ds)
  if len(sampled_xy) < 3:
    return sampled_xy, sampled_distances, 300.0, next_start_index, closest_point, []

  smooth_xy = smooth_polyline(sampled_xy, window_size=smooth_window)
  curvatures = compute_multiscale_curvature(smooth_xy, ds=ds, spans=(2, 4, 8))
  turn_angles = compute_local_turn_angle(smooth_xy, ds=ds, radius_m=18.0)

  raw_speeds = curvature_to_speed_kph(
    curvatures=curvatures,
    turn_angles_deg=turn_angles,
    vmax_kph=vmax_kph,
    a_lat_curve=1.8,
    turn_angle_soft=35.0,
    turn_angle_mid=50.0,
    turn_angle_hard=70.0,
    turn_speed_soft_kph=35.0,
    turn_speed_mid_kph=24.0,
    turn_speed_hard_kph=18.0,
  )

  final_speeds = apply_backward_speed_limit(
    speeds_kph=raw_speeds,
    ds=ds,
    decel_mps2=decel_mps2,
  )

  current_speed = final_speeds[0] if final_speeds else 300.0
  return smooth_xy, sampled_distances, current_speed, next_start_index, closest_point, final_speeds
