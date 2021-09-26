from datetime import timedelta, datetime, date

import six
from dateutil.parser import parse

MON, TUE, WED, THU, FRI, SAT, SUN = range(7)
WEEKEND = (SAT, SUN)

JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC = range(1, 13)


class HolidayBase(dict):
    PROVINCES = []

    def __init__(
        self, years=[], expand=True, observed=True, prov=None, state=None
    ):
        self.observed = observed
        self.expand = expand
        if isinstance(years, int):
            years = [
                years,
            ]
        self.years = set(years)
        if not getattr(self, "prov", False):
            self.prov = prov
        self.state = state
        for year in list(self.years):
            self._populate(year)

    def __setattr__(self, key, value):
        if key == "observed" and len(self) > 0:
            dict.__setattr__(self, key, value)
            if value is True:
                # Add (Observed) dates
                years = list(self.years)
                self.years = set()
                self.clear()
                for year in years:
                    self._populate(year)
            else:
                # Remove (Observed) dates
                for k, v in list(self.items()):
                    if v.find("Observed") >= 0:
                        del self[k]
        else:
            return dict.__setattr__(self, key, value)

    def __keytransform__(self, key):
        if isinstance(key, datetime):
            key = key.date()
        elif isinstance(key, date):
            key = key
        elif isinstance(key, int) or isinstance(key, float):
            key = datetime.utcfromtimestamp(key).date()
        elif isinstance(key, six.string_types):
            try:
                key = parse(key).date()
            except (ValueError, OverflowError):
                raise ValueError("Cannot parse date from string '%s'" % key)
        else:
            raise TypeError("Cannot convert type '%s' to date." % type(key))

        if self.expand and key.year not in self.years:
            self.years.add(key.year)
            self._populate(key.year)
        return key

    def __contains__(self, key):
        return dict.__contains__(self, self.__keytransform__(key))

    def __getitem__(self, key):
        if isinstance(key, slice):
            if not key.start or not key.stop:
                raise ValueError("Both start and stop must be given.")

            start = self.__keytransform__(key.start)
            stop = self.__keytransform__(key.stop)

            if key.step is None:
                step = 1
            elif isinstance(key.step, timedelta):
                step = key.step.days
            elif isinstance(key.step, int):
                step = key.step
            else:
                raise TypeError(
                    "Cannot convert type '%s' to int." % type(key.step)
                )

            if step == 0:
                raise ValueError("Step value must not be zero.")

            date_diff = stop - start
            if date_diff.days < 0 <= step or date_diff.days >= 0 > step:
                step *= -1

            days_in_range = []
            for delta_days in range(0, date_diff.days, step):
                day = start + timedelta(days=delta_days)
                try:
                    dict.__getitem__(self, day)
                    days_in_range.append(day)
                except KeyError:
                    pass
            return days_in_range
        return dict.__getitem__(self, self.__keytransform__(key))

    def __setitem__(self, key, value):
        if key in self:
            if self.get(key).find(value) < 0 and value.find(self.get(key)) < 0:
                value = "%s, %s" % (value, self.get(key))
            else:
                value = self.get(key)
        return dict.__setitem__(self, self.__keytransform__(key), value)

    def update(self, *args):
        args = list(args)
        for arg in args:
            if isinstance(arg, dict):
                for key, value in list(arg.items()):
                    self[key] = value
            elif isinstance(arg, list):
                for item in arg:
                    self[item] = "Holiday"
            else:
                self[arg] = "Holiday"

    def append(self, *args):
        return self.update(*args)

    def get(self, key, default=None):
        return dict.get(self, self.__keytransform__(key), default)

    def get_list(self, key):
        return [h for h in self.get(key, "").split(", ") if h]

    def get_named(self, name):
        # find all dates matching provided name (accepting partial
        # strings too, case insensitive), returning them in a list
        original_expand = self.expand
        self.expand = False
        matches = [key for key in self if name.lower() in self[key].lower()]
        self.expand = original_expand
        return matches

    def pop(self, key, default=None):
        if default is None:
            return dict.pop(self, self.__keytransform__(key))
        return dict.pop(self, self.__keytransform__(key), default)

    def pop_named(self, name):
        to_pop = self.get_named(name)
        if not to_pop:
            raise KeyError(name)
        for key in to_pop:
            self.pop(key)
        return to_pop

    def __eq__(self, other):
        return dict.__eq__(self, other) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return dict.__ne__(self, other) or self.__dict__ != other.__dict__

    def __add__(self, other):
        if isinstance(other, int) and other == 0:
            # Required to sum() list of holidays1
            # sum([h1, h2]) is equivalent to (0 + h1 + h2)
            return self
        elif not isinstance(other, HolidayBase):
            raise TypeError()
        HolidaySum = createHolidaySum(self, other)
        country = getattr(self, "country", None) or getattr(
            other, "country", None
        )
        if self.country and other.country and self.country != other.country:
            c1 = self.country
            if not isinstance(c1, list):
                c1 = [c1]
            c2 = other.country
            if not isinstance(c2, list):
                c2 = [c2]
            country = c1 + c2
        prov = getattr(self, "prov", None) or getattr(other, "prov", None)
        if self.prov and other.prov and self.prov != other.prov:
            p1 = self.prov if isinstance(self.prov, list) else [self.prov]
            p2 = other.prov if isinstance(other.prov, list) else [other.prov]
            prov = p1 + p2
        return HolidaySum(
            years=(self.years | other.years),
            expand=(self.expand or other.expand),
            observed=(self.observed or other.observed),
            country=country,
            prov=prov,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def _populate(self, year):
        pass

    def __reduce__(self):
        return super(HolidayBase, self).__reduce__()


def createHolidaySum(h1, h2):
    class HolidaySum(HolidayBase):
        def __init__(self, country, **kwargs):
            self.country = country
            self.holidays = []
            if getattr(h1, "holidays1", False):
                for h in h1.holidays:
                    self.holidays.append(h)
            else:
                self.holidays.append(h1)
            if getattr(h2, "holidays1", False):
                for h in h2.holidays:
                    self.holidays.append(h)
            else:
                self.holidays.append(h2)
            HolidayBase.__init__(self, **kwargs)

        def _populate(self, year):
            for h in self.holidays[::-1]:
                h._populate(year)
                self.update(h)

    return HolidaySum


class Russia(HolidayBase):
    """
    https://en.wikipedia.org/wiki/Public_holidays_in_Russia
    """

    def __init__(self, **kwargs):
        self.country = "RU"
        HolidayBase.__init__(self, **kwargs)

    def _populate(self, year):
        self[date(year, DEC, 31)] = "Пред. Новый год"
        # New Year's Day
        self[date(year, JAN, 1)] = "Новый год"
        # New Year's Day
        self[date(year, JAN, 2)] = "Новый год"
        # New Year's Day
        self[date(year, JAN, 3)] = "Новый год"
        # New Year's Day
        self[date(year, JAN, 4)] = "Новый год"
        # New Year's Day
        self[date(year, JAN, 5)] = "Новый год"
        # New Year's Day
        self[date(year, JAN, 6)] = "Новый год"
        # Christmas Day (Orthodox)
        self[date(year, JAN, 7)] = "Новый год"
        # New Year's Day
        self[date(year, JAN, 8)] = "Новый год"
        self[date(year, JAN, 9)] = "Новый год"
        # Man Day
        self[date(year, FEB, 23)] = "День защитника отечества"
        # Women's Day
        self[date(year, MAR, 8)] = "День женщин"
        # Labour Day
        self[date(year, MAY, 1)] = "Праздник Весны и Труда"
        self[date(year, MAY, 2)] = "Майские"
        self[date(year, MAY, 3)] = "Майские"
        # Victory Day
        self[date(year, MAY, 9)] = "День Победы"
        # Russia's Day
        self[date(year, JUN, 12)] = "День России"
        if year >= 2005:
            # Unity Day
            self[date(year, NOV, 4)] = "День народного единства"
        else:
            # October Revolution Day
            self[date(year, NOV, 7)] = "День Октябрьской революции"


class RU(Russia):
    pass


class RUS(Russia):
    pass
