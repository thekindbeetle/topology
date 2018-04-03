import numpy as np
import skimage.morphology
import skimage.filters


def binary_jacobi_set(field1, field2, threshold=0.7):
  """
    threshold — порог бинаризации.
  """
  field1 = field1 / field1.sum() * field1.size
  field2 = field2 / field2.sum() * field2.size
  # Считаем градиенты в каждой точке
  grad1 = np.dstack((skimage.filters.sobel_h(field1), skimage.filters.sobel_v(field1)))
  grad2 = np.dstack((skimage.filters.sobel_h(field2), skimage.filters.sobel_v(field2)))
  # Считаем логарифм градиентной меры
  grad = np.log(np.abs(np.cross(grad1, grad2)))
  # Находим множество Якоби
  jac = skimage.filters.sobel(grad)
  # Чистим его от всякой херни
  jac[np.isnan(jac)] = 0
  jac = (jac > threshold)
  jac = skimage.morphology.erosion(skimage.morphology.erosion(skimage.morphology.dilation(jac)))
  # Делаем линии тоньше
  jac = skimage.morphology.skeletonize(jac)
  return jac