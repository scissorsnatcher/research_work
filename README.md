## Установка библиотек

Не считая встроенных os и time, требуются библиотеки numpy, pandas и opencv.

Вариант установки: команда `pip install -r requirements.txt`

## Глобальные переменные

В начале скрипта есть блок "Globals", в котором:

- src_folder_name - имя папки с кадрами .bt8, она должна быть расположена там же, где и скрипт;

- f_number - число файлов, из которых усреднением получаем начальную картинку;

- P - доля нового кадра при изменении картинки (параметр экспоненциального скользящего среднего);

- P_scale - множитель P, используется при расширении контрастированного окна: увеличивает долю нового кадра;

- p_threshold - порог для бинаризации, доля от среднего по контрастированному окну; 

- bordr_min - минимальное расстояние от границы окна до пятна, также число пикселей, на которое расширяется/сужается окно;

- bordr_max - максимальное расстояние до границы окна;

- start_pres_key - код клавиши для начала режима презентации (изменение кадра во времени; по умолчанию _Enter_);

- close_wins_key - код клавиши для завершения работы скрипта (по умолчанию _Esc_).

## Работа скрипта

Сначала скрипт загружает и интерпретирует все кадры из указанной папки.

Далее открываются 4 окна: основное изображение (_Display_) и три малых, первое из которых показывает выделенную часть изображения (_dispROI_), во втором контрастированное изображение (_rectROI_), в третьем бинаризованное (_binROI_). Бинаризованное также дублируется в _Display_, поэтому выделенная область показывается в отдельном окне.

<details><summary>Как завершить работу скрипта</summary>
Начиная с этого этапа уже можно завершить работу скрипта клавишей Esc или закрытием окна Display.
</details>

Чтобы посмотреть, как кадры изменяются во времени, можно нажать клавишу _Enter_ - это запустит режим презентации.

Считанные из файлов кадры начнут изменять основное изображение по экспоненциальному скользящему среднему.

При этом, если пятно будет смещаться в окне, то размеры окна будут изменяться в том же направлении.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
