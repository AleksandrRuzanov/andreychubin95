import math as m
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame


class StorageCalculator:
    """
    Калькулятор стоимости палетоместа на основе ограниченных данных\n
    Цель - определение оптимальной конфигурации склада\n
    Процесс запускается методом self.annual_cost()\n

    :return датафрейм с итогами рассчётов и дополнительными метриками
    """

    def __init__(self,
                 input_file: DataFrame,
                 s: float,
                 height: float,
                 h: float,
                 period: int = 1,
                 horizon: int = 5,
                 tech: str = None,
                 max_levels: int = 5,
                 storage_utility: float = 0.05,
                 conversion: float = 0.33,
                 floor_conversion: float = 0.6,
                 workers_yearly_salary: float = 900000.0,
                 worker_coeff: float = 0.6,
                 worker_round_distance: float = 0.01,
                 sectorization: bool = False,
                 sector_area: float = 500.0):
        """
        :param input_file: датафрейм с ценами и стоймостями
        :param s: площадь склада (м2)
        :param height: высота потолка (м)
        :param h: высота межярусного расстояния
        :param period: период оборачиваемости (среднее время нахождения ТМЦ на складе) в месяцах (default=1)
        :param horizon: срок службы техники и стеллажей в года (default=5)
        :param tech: 'штабелёр' или 'погрузчик' (default 'штабелёр')
        :param max_levels: допустимый максимальный ярус стеллажа (default=7)
        :param storage_utility: коэффициент утилизации палетоместа
        :param conversion: коэффициент конверсии площади в количество мест стеллажного хранения
        :param floor_conversion: коэффициент конверсии площади в количество мест напольного хранения
        :param workers_yearly_salary: норматив годового оклада кладовщика/подсобного рабочего
        :param worker_coeff: коэффициент для пересчёта ПШЕ на площадь в ПШЕ на паллетоместо (исходя из 1 ПШЕ на 450 м2)
        :param worker_round_distance: дистанция округления для ПШЕ
        :param sectorization: включение/выключение функции секторизации
        (использовать только с правильным подбором размера сектора)
        :param sector_area: площадь сектора склада
        """

        self.input_file = input_file
        self.s = s
        self.height = height
        self.h = h
        self.tech = tech
        self.period = period
        self.horizon = horizon
        self.storage_utility = storage_utility
        self.conversion = conversion
        self.floor_conversion = floor_conversion
        self.workers_salary = workers_yearly_salary
        self.worker_coeff = worker_coeff
        self.worker_round_distance = worker_round_distance
        self.sectorization = sectorization
        self.sector_area = sector_area

        if max_levels > self.height // self.h:
            self.max_level = int(self.height // self.h)
        else:
            self.max_level = max_levels

    def set_default_tech(self) -> None:
        if self.max_level > 1 and self.tech is None:
            self.tech = 'штабелёр'

    def input_to_dict(self) -> dict:
        """
        конвертер датафрейма в объект dict (только для использования внутри класса)
        """
        dictionary = self.input_file.copy()
        dictionary.set_index(['Позиция'], drop=True, inplace=True)
        dictionary = dictionary.to_dict()
        dictionary = dictionary['Цена']

        return dictionary

    def annual_value(self, P, mean_w) -> float:
        """
        Примерное определение годового товарооборота скалада в тоннах\n
        :param P: количество палетомест на складе
        :param mean_w: средний вес одного пллета с грузом (кг)
        :return: int or float
        """
        mult = 12 // self.period
        real_P = (1 - self.storage_utility) * P

        return (real_P * mean_w * 2 * mult) / 1000
    
    @property
    def __forklift_trail_length(self) -> float:
        """
        Подсчёт среднего расстояния одного маршрута погрузчика на складе (с секторизацией)\n
        :return: float
        """
        result = 0
        if self.sectorization:
            c = self.s // self.sector_area
            if c <= 1:
                result = (2 * self.s) ** 0.5
            elif c >= 2:
                a = c - 1
                b = self.s - (a * self.sector_area)
                result = a * ((2 * self.sector_area) ** 0.5) + ((2 * b) ** 0.5)
        else:
            result = (2 * self.s) ** 0.5
            
        return result

    def __fork_machine_count(self, Q: float,
                           H_1: float,
                           alpha: float = 0.85,
                           T: int = 260,
                           q_f: float = 0.5,
                           q_n: float = 1.5,
                           t_1: float = 0.15,
                           t_0: float = 1.2,
                           V_0: float = 12.0,
                           V_1: float = 100.0,
                           T_day: float = 8) -> int:
        """
        Подсчёт требуемого количества погрузочной техники\n

        :param Q: годовой товарооборот в тоннах
        :param H_1: средняя высота подъёма вилки
        :param alpha: коэффициент по типу техники
        :param T: количество рабочих дней в году
        :param q_f: фактическая загрузка механизма (примерный средний вес перемещаемого ТМЦ) (т)
        :param q_n: номинальная грузоподъемность механизма (т)
        :param t_1: время, затрачиваемое на подъём рамы электропогрузчика, мин
        :param t_0: среднее время, затрачиваемое на выполнение вспомогательных операций, мин
        :param V_0: скорость подъема груза, м/мин
        :param V_1: скорость передвижения погрузчика, м/мин
        :param T_day: количество часов, которое работает склад в сутки
        :return: int
        """
        length = self.__forklift_trail_length
        a_gr = q_f / q_n
        t_loop = ((2.1 * H_1) / V_0) + ((2 * length) / V_1) + (4 * t_1) + t_0
        Q_hour = (60 * q_n * a_gr * alpha) / t_loop
        Q_day = Q / T
        fork = Q_day / (Q_hour * T_day)  # штуки

        if not self.sectorization:
            return m.ceil(fork)
        else:
            if fork < 1 and self.s // self.sector_area <= 1:
                return 1
            elif fork > 1 and m.ceil(fork) >= self.s // self.sector_area:
                return m.ceil(fork)
            else:
                return int(self.s // self.sector_area)

    def __workers_count(self, storage_places: list) -> list:
        """
        Подсчёт требуемого количества ПШЕ в зависимомти от количества палетомест\n
        :param storage_places: количество палетомест на складе (список)
        :return: список требуемого количества ПШЕ
        """
        worker_amount = []

        for i in range(0, len(storage_places)):
            a = storage_places[i] / (450 * self.worker_coeff)
            b = storage_places[i] // (450 * self.worker_coeff)
            if b == 0:
                worker_amount.append(1)
            else:
                c = a / b
                if c >= self.worker_round_distance:
                    worker_amount.append(int(m.ceil(a)))
                else:
                    worker_amount.append(int(a // 1))

        return worker_amount

    @property
    def __cost_of_storage_place_purchase(self) -> DataFrame:
        """
        Рассчёт стоимости обустройства склада
        :return: датафрейм с результатми расчётов стоимости приобретения палетоместа и параметрами
        """
        self.set_default_tech()
        dictionary = self.input_to_dict()
        block_amount = (self.conversion * self.s) // 3
        floor_storage_amount = (self.floor_conversion * self.s)
        rochlya = dictionary.get('Цена рохли')
        pogr_2200 = dictionary.get('Цена погрузчика с вилкой 2200мм')
        pogr_5000 = dictionary.get('Цена погрузчика с вилкой 5000мм')
        pogr_6000 = dictionary.get('Цена погрузчика с вилкой 6000мм')
        pogr_high = dictionary.get('Цена высотного погрузчика')
        stab_2200 = dictionary.get('Цена штабелёра с вилкой 2200мм')
        stab_5000 = dictionary.get('Цена штабелёра с вилкой 5000мм')
        stab_6000 = dictionary.get('Цена штабелёра с вилкой 6000мм')
        levels = range(1, self.max_level + 1)
        base_storage_places = block_amount * 3
        storage_places = [base_storage_places * x if x != 1 
                          else floor_storage_amount for x in levels]
        workers_amount = self.__workers_count(storage_places)
        workers_cost_per_year = [workers_amount[w] * self.workers_salary for w in range(0, len(workers_amount))]
        workers_cost_storage_place = [round(workers_cost_per_year[w] / storage_places[w], 2)
                                      for w in range(0, len(workers_cost_per_year))]

        tech_price = []
        list_q = []
        tech_amount = []
        add_rochlya = []

        def helper_function(tier: int, tech_1: float = None, tech_2: float = None) -> None:
            """
            эта функция помогает не писать один и тот же код много раз
            """
            Q = self.annual_value(P=base_storage_places * tier, mean_w=400)
            n = self.__fork_machine_count(Q=Q, H_1=tier)
            
            if tech_1 is not None:
                tech_price.append(tech_1)
            else:
                tech_price.append((tech_2 * n) + (rochlya * (workers_amount[tier - 1] - n)))
                
            list_q.append(Q)
            tech_amount.append(n)
            add_rochlya.append(workers_amount[tier - 1] - n)

        for lev in levels:
            if lev == 1:
                helper_function(tier=lev, tech_1=rochlya * workers_amount[lev - 1])
            else:
                if self.tech == 'погрузчик':
                    if lev == 2:
                        helper_function(tier=lev, tech_2=pogr_2200)
                    elif lev == 3:
                        helper_function(tier=lev, tech_2=pogr_5000)
                    elif lev == 4:
                        helper_function(tier=lev, tech_2=pogr_6000)
                    elif lev >= 5:
                        helper_function(tier=lev, tech_2=pogr_high)

                elif self.tech == 'штабелёр':
                    if lev == 2:
                        helper_function(tier=lev, tech_2=stab_2200)
                    elif lev == 3:
                        helper_function(tier=lev, tech_2=stab_5000)
                    elif lev == 4:
                        helper_function(tier=lev, tech_2=stab_6000)
                    elif lev >= 5:
                        helper_function(tier=lev, tech_2=pogr_high)

        stellage_comp_cost = dictionary.get('Доп. расходы на стеллаж за секцию (монтаж, отбойники и тд.)') \
                             * block_amount
        stellage_base_cost = dictionary.get('Цена первого яруса стеллажа (пол+1) за секцию') \
                             * block_amount
        stellage_level_cost = dictionary.get(
            'Цена одного дополнительного яруса стеллажа (за одну секцию 2700*1100*2000)') * block_amount
        stellage_cost = [stellage_comp_cost + stellage_base_cost + (stellage_level_cost * (x - 2))
                         if x != 1 else 0 for x in levels]
        storage_cost_per_place = [round((tech_price[index] + stellage_cost[index]) / storage_places[index])
                                  for index in range(0, len(tech_price))]
        volume = [self.annual_value(P=storage_places[place], mean_w=0.7 * 1000)
                  for place in range(0, len(storage_places))]
        values = [levels, storage_places, tech_amount, add_rochlya,
                  storage_cost_per_place, list_q, volume,
                  workers_amount, workers_cost_storage_place]

        columns = ['Количество ярусов',
                   'Количество палетомест',
                   'Количество единиц техники',
                   'Количество дополнительных рохлей',
                   'Стоимость палетоместа при покупке (руб.)',
                   'Возможный товарооборот (т/год)',
                   'Возможный товарооборот (куб. м/год)',
                   'Требуемое количество ПШЕ',
                   'Расходы на персонал (руб/пм в год)']

        dataframe = pd.DataFrame()
        
        for value, column in zip(values, columns):
            dataframe[column] = value

        def tech_adder(num):
            if num == 1:
                result = 'рохля'
            elif num in [2, 3, 4] and self.tech != 'погрузчик':
                result = 'штабелёр'
            elif num in [2, 3, 4] and self.tech == 'погрузчик':
                result = 'погрузчик'
            else:
                result = 'ричтрак'
                
            return result
        
        dataframe['Тип техники'] = dataframe['Количество ярусов'].apply(tech_adder)

        return dataframe

    @property
    def __annual_cost_of_maintenance_per_storage_place(self) -> list:
        """
        Рассчёт стоимости содержания в год на палетоместо
        :return: стоимость обслуживания палетоместа в год (list)
        """
        dictionary = self.input_to_dict()
        data_frame = self.__cost_of_storage_place_purchase
        maintenance_cost_per_sq_meter = \
            dictionary.get('Средняя стоимость содержания склада за кв. метр (аренда + обслуживание в месяц)')
        maintenance_cost = maintenance_cost_per_sq_meter * self.s * 12
        maintenance = (round(maintenance_cost / data_frame['Количество палетомест'], 2)).tolist()

        return maintenance

    def annual_cost(self) -> DataFrame:
        """
        Рассчёт общей стоимости на год за палетоместо
        :return: стоимость содержания палетоместа в год (list)
        """

        data_frame = self.__cost_of_storage_place_purchase
        maintenance = self.__annual_cost_of_maintenance_per_storage_place

        data_frame[f'Стоимость приобретения (руб/пм в год) (горизонт {self.horizon} лет)'] = \
            round(data_frame['Стоимость палетоместа при покупке (руб.)'] / self.horizon, 2)

        data_frame['Стоимость обслуживания (руб/пм в год)'] = maintenance

        data_frame['Годовая стоимость за пм'] = \
            data_frame[f'Стоимость приобретения (руб/пм в год) (горизонт {self.horizon} лет)'] + \
            data_frame['Стоимость обслуживания (руб/пм в год)'] + \
            data_frame['Расходы на персонал (руб/пм в год)']

        data_frame['Общая годовая стоимость (млн. руб)'] = round(
            data_frame['Годовая стоимость за пм'] *
            (data_frame['Количество палетомест'] / 1000000), 4
        )

        # расставляю колонки в правильном порядке
        columns = ['Количество ярусов',
                   'Количество палетомест',
                   'Количество единиц техники',
                   'Тип техники',
                   'Количество дополнительных рохлей',
                   'Стоимость палетоместа при покупке (руб.)',
                   'Возможный товарооборот (т/год)',
                   'Возможный товарооборот (куб. м/год)',
                   'Требуемое количество ПШЕ',
                   'Расходы на персонал (руб/пм в год)',
                   'Стоимость приобретения (руб/пм в год) (горизонт ' + str(self.horizon) + ' лет)',
                   'Стоимость обслуживания (руб/пм в год)',
                   'Годовая стоимость за пм',
                   'Общая годовая стоимость (млн. руб)']

        return data_frame[columns]

    def short_run(self, places_to_volume: float = 0.80) -> (int, float):
        data_frame = self.__cost_of_storage_place_purchase
        data_frame = data_frame[['Количество палетомест']].copy()
        data_frame['Куб. метров паллетного хранения'] = data_frame['Количество палетомест'].apply(
            lambda x: round(places_to_volume * x, 2))

        return data_frame['Количество палетомест'].max(), data_frame['Куб. метров паллетного хранения'].max()


def optimal_area_per_level(input_file: DataFrame,
                           list_of_areas: list,
                           h: float,
                           horizon: int = 5) -> (DataFrame, DataFrame, DataFrame, DataFrame, DataFrame):
    """
    Рассчёт трат в разрезе различных площадей и различной ярусности\n
    :param input_file: DataFrame со стоимостями
    :param list_of_areas: список площадей, по которым нужно провести анализ
    :param h: высота межярусного расстояния
    :param horizon: горизонт амортизации в годах (default 5)
    :return: 5 DataFrame'ов
    """
    calc_base = StorageCalculator(input_file=input_file, s=100, height=20, h=h, horizon=horizon)
    data_frame = calc_base.annual_cost()
    data_frame.drop(data_frame.index, inplace=True)

    data_frame1 = data_frame.copy()
    data_frame2 = data_frame.copy()
    data_frame3 = data_frame.copy()
    data_frame4 = data_frame.copy()
    data_frame5 = data_frame.copy()

    for element in list_of_areas:
        calc = StorageCalculator(input_file=input_file, s=element, height=999, h=h)
        data_frame_1 = calc.annual_cost()

        data_frame1 = data_frame1.append(data_frame_1[data_frame_1.index == 0], ignore_index=True)
        data_frame2 = data_frame2.append(data_frame_1[data_frame_1.index == 1], ignore_index=True)
        data_frame3 = data_frame3.append(data_frame_1[data_frame_1.index == 2], ignore_index=True)
        data_frame4 = data_frame4.append(data_frame_1[data_frame_1.index == 3], ignore_index=True)
        data_frame5 = data_frame5.append(data_frame_1[data_frame_1.index == 4], ignore_index=True)

    for data_frame_iter in [data_frame1, data_frame2, data_frame3, data_frame4, data_frame5]:
        data_frame_iter['Площадь'] = list_of_areas

    return data_frame1, data_frame2, data_frame3, data_frame4, data_frame5


def visualization_area(input_file: DataFrame,
                       list_of_areas: list,
                       h: float,
                       save: bool = False) -> None:
    """
    Визуализация результатов рассчёта функции optimal_area_per_level()\n
    :param input_file: DataFrame со стоимостями
    :param list_of_areas: список площадей, по которым нужно провести анализ
    :param h: высота межярусного расстояния
    :param save: включени/отключение функции сохранения результата в виде png
    :return: график (+ png)
    """
    data_frame1, data_frame2, data_frame3, data_frame4, data_frame5 = \
        optimal_area_per_level(input_file, list_of_areas, h)

    plt.figure(figsize=(20, 12))

    plt.subplot(221)
    plt.plot(data_frame1['Площадь'], data_frame1['Годовая стоимость за пм'],
             data_frame2['Площадь'], data_frame2['Годовая стоимость за пм'],
             data_frame3['Площадь'], data_frame3['Годовая стоимость за пм'],
             data_frame4['Площадь'], data_frame4['Годовая стоимость за пм'],
             data_frame5['Площадь'], data_frame5['Годовая стоимость за пм'])
    plt.legend(['Пол', 'Пол+1', 'Пол+2', 'Пол+3', 'Пол+4'])
    plt.xlabel('Площадь')
    plt.ylabel('Годовая стоимость за пм')

    plt.subplots_adjust(wspace=0.4)

    if save:
        plt.savefig('pic.png', dpi=500, bbox_inches='tight')

    plt.show()


def visualization(data_frame: DataFrame, save: bool = False) -> None:
    """
    Визуализация данных\n
    :param data_frame: результат работы класса StorageCalculator
    :param save: сохранить результат в виде картинки?
    :return: 4 графика
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.plot(data_frame['Количество ярусов'], data_frame['Годовая стоимость за пм'])
    plt.xlabel('Количество ярусов')
    plt.ylabel('Годовая стоимость за пм')

    plt.subplot(222)
    plt.plot(data_frame['Количество ярусов'], data_frame['Возможный товарооборот (куб. м/год)'])
    plt.xlabel('Количество ярусов')
    plt.ylabel('Возможный товарооборот (куб. м/год)')

    plt.subplot(223)
    plt.plot(data_frame['Количество ярусов'], data_frame['Количество единиц техники'],
             data_frame['Количество ярусов'], data_frame['Требуемое количество ПШЕ'])
    plt.legend(['Техника', 'ПШЕ'])
    plt.xlabel('Количество ярусов')
    plt.ylabel('Количество требуемых единиц')

    plt.subplot(224)
    plt.plot(data_frame['Количество ярусов'], data_frame['Общая годовая стоимость (млн. руб)'])
    plt.xlabel('Количество ярусов')
    plt.ylabel('Общая годовая стоимость (млн. руб)')

    plt.subplots_adjust(wspace=0.4)

    if save:
        plt.savefig('picture.png', dpi=500, bbox_inches='tight')

    plt.show()
