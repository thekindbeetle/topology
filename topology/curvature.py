import numpy as np
import skimage.morphology
import skimage.filters
import itertools


# Фильтр границы окрестности
_border_filter = np.array(
  [[1, 1, 1, 1, 1],
   [1, 0, 0, 0, 1],
   [1, 0, 0, 0, 1],
   [1, 0, 0, 0, 1],
   [1, 1, 1, 1, 1]]
)


# Карта расстояний до центра в окрестности
_distance_map = np.zeros((5, 5))
for i in range(5):
  for j in range(5):
    _distance_map[i, j] = np.linalg.norm((2 - i, 2 - j))

        
# Угол, соответствующий каждой из точек окрестности.
_angles = np.zeros((5, 5))
for i in range(5):
  for j in range(5):
    _angles[i, j] = np.arctan2(2 - i, 2 - j)

        
def _get_angle(coord1, coord2):
  p1, p2 = tuple(coord1), tuple(coord2)
  angle = np.abs(_angles[p1] - _angles[p2])
  if angle > np.pi:
    angle = 2 * np.pi - angle
  return angle

    
def get_curvature_at_point(field, x, y):
  if field[x, y] == 0:
    return 0
      
  # Рассмотрим окрестность 5 x 5
  window = field[x - 2: x + 3, y - 2: y + 3]
  
  # Рассмотрим значения на границе
  border = _border_filter & window
  coords = np.transpose(np.where(border == 1))
  
  # Если точек на границе не две
  if len(coords) > 2:
    # Выберем две, образующие максимальный угол
    pairs = list(itertools.combinations(coords, 2))
    angles_list = list(map(lambda p: _get_angle(p[0], p[1]), pairs))
    coords = pairs[np.argmax(angles_list)]
  elif len(coords) < 2:
    return 0
  
  in_point, out_point = tuple(map(tuple, coords))
  angle = _get_angle(in_point, out_point)
  curvature = 2 * (np.pi - angle) / (_distance_map[in_point] + _distance_map[out_point])
  return curvature
    
def get_curvature_map(field):
  f = field.copy()
  # Продлеваем поле по краям
  f = np.vstack([f[-2:,:], f, f[:2,:]])
  f = np.hstack([f[:, -2:], f, f[:, :2]])
  curvature_map = np.zeros(f.shape)
  for i in range(2, curvature_map.shape[0] - 2):
    for j in range(2, curvature_map.shape[1] - 2):
      curvature_map[i, j] = get_curvature_at_point(f, i, j)
  # Теперь обрезаем лишнее.
  return curvature_map[2:-2]
  
def get_mean_curvature(field):
  bool_field = (field > 0)
  return np.sum(get_curvature_map(field)) / np.sum(bool_field)
  