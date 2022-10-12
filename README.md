# ```topology``` — пакет для моделирования и обработки дискретных данных (сеток, триангуляций и наборов точек).

## Как с ним работать без IDE?

Клонируем репозиторий:

```hg clone <адрес репозитория> local-directory```

Добавляем адрес репозитория в переменную среды ```PYTHONPATH```

Далее, может требоваться установка пакетов python. Так, **точно** понадобятся пакеты ```numpy```, ```scipy```, ```matplotlib```, ```triangle```.

Для возможности работы с ```R```, нужно установить пакет ```rpy2```. При этом, переменная среды ```R_HOME``` должна указывать на директорию установки ```R```.

## Реализованные алгоритмы:

* ```topology.morse``` Вычисление дискретного комплекса Морса-Смейла. Алгоритм построения дискретного градиента взят из статьи[^2], склейка 2D по тору; алгоритм выделения сепаратрисвзят отсюда[^1].
* ```topology.morse``` Топологическое упрощение комплекса Морса-Смейла. См.[^1], в простой версии происходит изменение дискретного градиента (так, чтобы на каждом шаге комплекс терял одну пару <<экстремум -- седло>>. Удаление <<усов>> при упрощении делается руками (когда вычисленная сепаратриса после разворота градиента содержит участки, проходимые по нескольку раз).
* ```topology.morse``` Восстановление функции по градиенту. Фактически, решается задача Дирихле: граничные условия (значения на сепаратрисах) известны, а дальше прогоняется метод конечных элементов на каждой ячейке комплекса. При желании, можно сгладить получившуюся картинку вдоль сепаратрис, но я так не делал.
* ```topology.morse``` Построение множества Якоби[^4]. Несколько вариантов реализации: ```jacobi_image```: тупой, но быстрый алгоритм вычисления по определению. Численно считаем градиенты, находим (с погрешностью), где они ортогональны.
* ```topology.morse.scalespacemesh3d```: алгоритм вычисления дискретного приближения множества Якоби в Scale Space[^5]. Выделяются топ-точки (точки вырождения гессиана) в Scale Space, на них строится граф, показывающий перемещение критических точек в зависимости от масштаба (уровня сглаживания).
* ```topology.point_process``` [Генераторы точечных процессов](https://hpaulkeeler.com/simulating-a-matern-cluster-point-process/): процесс Пуассона, процесс Матерна и процесс Томаса
* ```topology.triangulation``` [Триангуляция Делоне](https://en.wikipedia.org/wiki/Delaunay_triangulation): реализация из пакета ```triangle```.
* ```topology.triangulation.persistence``` Вычисление чисел Бетти и персистентных гомологий[^3] для триангуляции с заданной в вершинах функцией.

[^1]: Sousbie T. Persistence cosmic web and its filamentary structure.
[^2]: Robins et al. Theory and algorithms for constructing discrete Morse complexes from grayscale digital images, IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 33, NO. 8, AUGUST
[^3]: Persistent Homology — a Survey. Herbert Edelsbrunner and John Harer.
[^4]: Jacobi Sets of Multiple Morse Functions. Herbert Edelsbrunner and John Harer.
[^5]: [Top-Points as Interest Points for Image Matching](https://www.researchgate.net/publication/221304150_Top-Points_as_Interest_Points_for_Image_Matching). Platel B. et al
