import math as m
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class StorageCalculator:
    """
    Калькулятор стоимости палетоместа на основе ограниченных данных\n
    Цель - определение оптимальной конфигурации склада\n
    Процесс запускается методом self.annual_cost()\n

    :return датафрейм с итогами рассчётов и дополнительными метриками
    """

    def __init__(self, input_file, s, height,
                 period=1, horizon=5,
                 tech=None, max_levels=5,
                 storage_utility=0.05,
                 conversion=0.33,
                 floor_conversion=0.6,
                 workers_yearly_salary=900000,
                 worker_coeff=0.6,
                 worker_round_distance=0.01,
                 sectorization=0,
                 sector_area=500):
        """
        :param input_file: датафрейм с ценами и стоймостями
        :param s: площадь склада (м2)
        :param height: высота потолка (м)
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
        :param sector_area: площадь сектора склада
        """

        self.input_file = input_file
        self.s = s
        self.height = height
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

        if max_levels > self.height // 2:
            self.max_level = int(self.height // 2)
        else:
            self.max_level = max_levels

    def setDefaultTech(self):
        if self.max_level > 1 and self.tech is None:
            self.tech = 'штабелёр'

    def input_to_dict(self):
        """
        конвертер датафрейма в объект dict (только для использования внутри класса)
        """
        dictionary = self.input_file.copy()
        dictionary.set_index(['Позиция'], drop=True, inplace=True)
        dictionary = dictionary.to_dict()
        dictionary = dictionary['Цена']

        return dictionary

    def annual_value(self, P, mean_w):
        """
        Примерное определение годового товарооборота скалада в тоннах\n
        :param P: количество палетомест на складе
        :param mean_w: средний вес одного пллета с грузом (кг)
        :return: int or float
        """
        mult = 12 // self.period
        real_P = (1 - self.storage_utility) * P

        return (real_P * mean_w * 2 * mult) / 1000

    def forklift_trail_length(self):
        """
        Подсчёт среднего расстояния одного маршрута погрузчика на складе (с секторизацией)\n
        :return: int or float
        """
        c = self.s // self.sector_area

        if c <= 1:
            return (2 * self.s) ** 0.5

        elif c >= 2:
            a = c - 1
            b = self.s - (a * self.sector_area)
            return a * ((2 * self.sector_area) ** 0.5) + ((2 * b) ** 0.5)

    def fork_machine_count(self, Q, H_1, alpha=0.85, T=260, q_f=0.5, q_n=1.5, t_1=0.15, t_0=1.2,
                           V_0=12, V_1=100, T_day=8):
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
        if self.sectorization == 1:
            l = self.forklift_trail_length()
        else:
            l = (2 * self.s) ** 0.5

        a_gr = q_f / q_n
        t_loop = ((2.1 * H_1) / V_0) + ((2 * l) / V_1) + (4 * t_1) + t_0
        Q_hour = (60 * q_n * a_gr * alpha) / t_loop
        Q_day = Q / T
        fork = Q_day / (Q_hour * T_day)  # штуки

        if fork < 1 and self.s // self.sector_area <= 1:
            return 1
        elif fork > 1 and m.ceil(fork) >= self.s // self.sector_area:
            return m.ceil(fork)
        else:
            return self.s // self.sector_area

    def workers_count(self, storage_places):
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

        def cost_of_storage_place_purchase(self):
        """
        Рассчёт стоимости обустройства склада
        :return: датафрейм с результатми расчётов стоимости приобретения палетоместа и параметрами
        """
        self.setDefaultTech()

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

        storage_places = []

        for num in levels:
            if num == 1:
                storage_places.append(floor_storage_amount)
            else:
                storage_places.append(base_storage_places * num)

        workers_amount = self.workers_count(storage_places)
        workers_cost_per_year = [workers_amount[w] * self.workers_salary for w in range(0, len(workers_amount))]
        workers_cost_storage_place = [round(workers_cost_per_year[w] / storage_places[w], 2)
                                      for w in range(0, len(workers_cost_per_year))]

        tech_price = []
        list_q = []
        tech_amount = []
        add_rochlya = []

        def helper_function(l, tech_1=None, tech_2=None):
            """
            эта функция помогает не писать один и тот же код много раз
            """
            Q = self.annual_value(P=base_storage_places * l, mean_w=400)
            n = self.fork_machine_count(Q=Q, H_1=l)

            if tech_1 is not None:
                tech_price.append(tech_1)
            else:
                tech_price.append((tech_2 * n) + (rochlya * (workers_amount[lev - 1] - n)))

            list_q.append(Q)
            tech_amount.append(n)
            add_rochlya.append(workers_amount[l - 1] - n)

        for lev in levels:
            if lev == 1:
                helper_function(l=lev, tech_1=rochlya * workers_amount[lev - 1])
            else:
                if self.tech == 'погрузчик':
                    if lev == 2:
                        helper_function(l=lev, tech_2=pogr_2200)
                    elif lev == 3:
                        helper_function(l=lev, tech_2=pogr_5000)
                    elif lev == 4:
                        helper_function(l=lev, tech_2=pogr_6000)
                    elif lev >= 5:
                        helper_function(l=lev, tech_2=pogr_high)

                elif self.tech == 'штабелёр':
                    if lev == 2:
                        helper_function(l=lev, tech_2=stab_2200)
                    elif lev == 3:
                        helper_function(l=lev, tech_2=stab_5000)
                    elif lev == 4:
                        helper_function(l=lev, tech_2=stab_6000)
                    elif lev >= 5:
                        helper_function(l=lev, tech_2=pogr_high)

        stellage_comp_cost = dictionary.get('Доп. расходы на стеллаж за секцию (монтаж, отбойники и тд.)') \
                             * block_amount
        stellage_base_cost = dictionary.get('Цена первого яруса стеллажа (пол+1) за секцию') \
                             * block_amount
        stellage_level_cost = dictionary.get(
            'Цена одного дополнительного яруса стеллажа (за одну секцию 2700*1100*2000)') * block_amount

        stellage_cost = []

        for level in levels:
            if level == 1:
                stellage_cost.append(0)
            else:
                stellage_cost.append(stellage_comp_cost + stellage_base_cost + (stellage_level_cost * (level - 2)))

        storage_cost_per_place = [round((tech_price[index] + stellage_cost[index]) / storage_places[index])
                                  for index in range(0, len(tech_price))]

        volume = [self.annual_value(P=storage_places[place], mean_w=0.7 * 1000)
                  for place in range(0, len(storage_places))]

        data = []

        for element in range(0, len(storage_cost_per_place)):
            a = [levels[element],
                 storage_places[element],
                 tech_amount[element],
                 add_rochlya[element],
                 storage_cost_per_place[element],
                 list_q[element],
                 volume[element],
                 workers_amount[element],
                 workers_cost_storage_place[element]]

            data.append(a)

        dataframe = pd.DataFrame(data, columns=['Количество ярусов',
                                                'Количество палетомест',
                                                'Количество единиц техники',
                                                'Количество дополнительных рохлей',
                                                'Стоимость палетоместа при покупке (руб.)',
                                                'Возможный товарооборот (т/год)',
                                                'Возможный товарооборот (куб. м/год)',
                                                'Требуемое количество ПШЕ',
                                                'Расходы на персонал (руб/пм в год)'])

        dataframe['Тип техники'] = ' '

        for row in dataframe.index:
            if dataframe.loc[row, 'Количество ярусов'] == 1:
                dataframe.at[row, 'Тип техники'] = 'рохля'
            elif dataframe.loc[row, 'Количество ярусов'] in [2, 3, 4] and self.tech != 'погрузчик':
                dataframe.at[row, 'Тип техники'] = 'штабелёр'
            elif dataframe.loc[row, 'Количество ярусов'] in [2, 3, 4] and self.tech == 'погрузчик':
                dataframe.at[row, 'Тип техники'] = 'погрузчик'
            else:
                dataframe.at[row, 'Тип техники'] = 'ричтрак'

        return dataframe

    def annual_cost_of_maintenance_per_storage_place(self):
        """
        Рассчёт стоимости содержания в год на палетоместо
        :return: стоимость обслуживания палетоместа в год (list)
        """

        dictionary = self.input_to_dict()
        df = self.cost_of_storage_place_purchase()

        maintenance_cost_per_sq_meter = \
            dictionary.get('Средняя стоимость содержания склада за кв. метр (аренда + обслуживание в месяц)')

        maintenance_cost = maintenance_cost_per_sq_meter * self.s * 12

        maintenance = []

        for i in df.index:
            maintenance.append(maintenance_cost / df.loc[i]['Количество палетомест'])

        return maintenance

    def annual_cost(self):
        """
        Рассчёт общей стоимости на год за палетоместо
        :return: стоимость содержания палетоместа в год (list)
        """

        df = self.cost_of_storage_place_purchase()
        maintenance = self.annual_cost_of_maintenance_per_storage_place()

        df['Стоимость приобретения (руб/пм в год) (горизонт ' + str(self.horizon) + ' лет)'] = \
            df['Стоимость палетоместа при покупке (руб.)'].apply(lambda x: round(x / self.horizon, 2))

        df['Стоимость обслуживания (руб/пм в год)'] = np.nan

        for ind in df.index:
            df.at[ind, 'Стоимость обслуживания (руб/пм в год)'] = round(maintenance[ind], 2)

        df['Годовая стоимость за пм'] = df['Стоимость приобретения (руб/пм в год) (горизонт ' + \
                                           str(self.horizon) + ' лет)'] + \
                                        df['Стоимость обслуживания (руб/пм в год)'] + \
                                        df['Расходы на персонал (руб/пм в год)']

        df['Общая годовая стоимость (млн. руб)'] = df['Годовая стоимость за пм'] * \
                                                   df['Количество палетомест'].apply(lambda x: x / 1000000)

        df['Общая годовая стоимость (млн. руб)'] = df['Общая годовая стоимость (млн. руб)'].apply(lambda x: round(x, 4))

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

        return df[columns]

    def short_run(self, PlacesToVolume=0.81):
        df = self.cost_of_storage_place_purchase()
        df = df[['Количество палетомест']].copy()
        df['Куб. метров паллетного хранения'] = df['Количество палетомест'].apply(
            lambda x: round(PlacesToVolume * x, 2))

        return df['Количество палетомест'].max(), df['Куб. метров паллетного хранения'].max()


def optimal_area_per_level(input_file, list_of_areas, h, horizon=5):
    """
    Рассчёт трат в разрезе различных площадей и различной ярусности\n
    :param input_file: DataFrame со стоимостями
    :param list_of_areas: список площадей, по которым нужно провести анализ
    :param h: высота межярусного расстояния
    :param horizon: горизонт амортизации в годах (default 5)
    :return: 5 DataFrame'ов
    """
    calc_base = StorageCalculator(input_file=input_file, s=100, height=20, h=h, horizon=horizon)
    df = calc_base.annual_cost()
    df.drop(df.index, inplace=True)

    df1 = df.copy()
    df2 = df.copy()
    df3 = df.copy()
    df4 = df.copy()
    df5 = df.copy()

    for element in list_of_areas:
        calc = StorageCalculator(input_file=input_file, s=element, height=999, h=h)
        df_1 = calc.annual_cost()

        df1 = df1.append(df_1[df_1.index == 0], ignore_index=True)
        df2 = df2.append(df_1[df_1.index == 1], ignore_index=True)
        df3 = df3.append(df_1[df_1.index == 2], ignore_index=True)
        df4 = df4.append(df_1[df_1.index == 3], ignore_index=True)
        df5 = df5.append(df_1[df_1.index == 4], ignore_index=True)

    for df_iter in [df1, df2, df3, df4, df5]:
        df_iter['Площадь'] = list_of_areas

    return df1, df2, df3, df4, df5


def visualization_area(input_file, list_of_areas, height=999, save=0):
    """
    Визуализация результатов рассчёта функции optimal_area_per_level()\n
    :param input_file: DataFrame со стоимостями
    :param list_of_areas: список площадей, по которым нужно провести анализ
    :param height: предельная высота потолка (определяет количество уровней) (set 999 for max 5 value)
    :param save: включени/отключение функции сохранения результата в виде png
    :return: график (+ png)
    """
    df1, df2, df3, df4, df5 = optimal_area_per_level(input_file, list_of_areas, height)

    plt.figure(figsize=(20, 12))

    plt.subplot(221)
    plt.plot(df1['Площадь'], df1['Годовая стоимость за пм'],
             df2['Площадь'], df2['Годовая стоимость за пм'],
             df3['Площадь'], df3['Годовая стоимость за пм'],
             df4['Площадь'], df4['Годовая стоимость за пм'],
             df5['Площадь'], df5['Годовая стоимость за пм'])
    plt.legend(['Пол', 'Пол+1', 'Пол+2', 'Пол+3', 'Пол+4'])
    plt.xlabel('Площадь')
    plt.ylabel('Годовая стоимость за пм')

    plt.subplots_adjust(wspace=0.4)

    if save == 1:
        plt.savefig('pic.png', dpi=500, bbox_inches='tight')

    plt.show()


def visualization(df, save=0):
    """
    Визуализация данных\n
    :param df: результат работы класса StorageCalculator
    :return: 4 графика
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.plot(df['Количество ярусов'], df['Годовая стоимость за пм'])
    plt.xlabel('Количество ярусов')
    plt.ylabel('Годовая стоимость за пм')

    plt.subplot(222)
    plt.plot(df['Количество ярусов'], df['Возможный товарооборот (куб. м/год)'])
    plt.xlabel('Количество ярусов')
    plt.ylabel('Возможный товарооборот (куб. м/год)')

    plt.subplot(223)
    plt.plot(df['Количество ярусов'], df['Количество единиц техники'],
             df['Количество ярусов'], df['Требуемое количество ПШЕ'])
    plt.legend(['Техника', 'ПШЕ'])
    plt.xlabel('Количество ярусов')
    plt.ylabel('Количество требуемых единиц')

    plt.subplot(224)
    plt.plot(df['Количество ярусов'], df['Общая годовая стоимость (млн. руб)'])
    plt.xlabel('Количество ярусов')
    plt.ylabel('Общая годовая стоимость (млн. руб)')

    plt.subplots_adjust(wspace=0.4)

    if save == 1:
        plt.savefig('picture.png', dpi=500, bbox_inches='tight')

    plt.show()
