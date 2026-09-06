"""对已经选定的证据做程序计算；不猜日期、单位换算关系或公共查询时间。"""

from __future__ import annotations

from calendar import month_abbr, month_name, monthrange
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import re


_PRECISION = {"year": 1, "month": 2, "day": 3, "minute": 3, "second": 3}
_MONTHS = {name.casefold(): number for number in range(1, 13)
           for name in (month_name[number], month_abbr[number])}
_MONTHS["sept"] = 9
_MONTH_WORDS = "|".join(sorted(_MONTHS, key=len, reverse=True))
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_DAY_FIRST = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_WORDS})\.?\s*,?\s*(\d{{4}})\b", re.I)
_MONTH_FIRST = re.compile(rf"\b({_MONTH_WORDS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})\b", re.I)
_MONTH_YEAR = re.compile(rf"\b({_MONTH_WORDS})\.?\s+(\d{{4}})\b", re.I)


@dataclass(frozen=True)
class _CalendarDate:
    year: int
    month: int | None = None
    day: int | None = None

    @property
    def precision(self):
        return "day" if self.day is not None else "month" if self.month is not None else "year"

    def calendar_day(self):
        if self.day is None:
            raise ValueError("Day-precision dates are required for this operation")
        return date(self.year, self.month, self.day)


def _parse_date(value: str) -> _CalendarDate:
    value = value.strip()
    if re.fullmatch(r"\d{4}", value):
        date(int(value), 1, 1)
        return _CalendarDate(int(value))
    if re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = map(int, value.split("-"))
        date(year, month, 1)
        return _CalendarDate(year, month)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        return _CalendarDate(parsed.year, parsed.month, parsed.day)
    except ValueError:
        for pattern, order in ((_MONTH_FIRST, "mdy"), (_DAY_FIRST, "dmy"), (_MONTH_YEAR, "my")):
            match = pattern.fullmatch(value)
            if match:
                parts = dict(zip(order, match.groups()))
                month, year = _MONTHS[parts["m"].casefold()], int(parts["y"])
                day = int(parts["d"]) if "d" in parts else None
                date(year, month, day or 1)
                return _CalendarDate(year, month, day)
        raise ValueError("Date operand must be an explicit ISO or unambiguous written calendar date") from None


def _public_date(query_time) -> _CalendarDate:
    if isinstance(query_time, datetime):
        query_time = query_time.date().isoformat()
    elif isinstance(query_time, date):
        query_time = query_time.isoformat()
    elif not isinstance(query_time, str):
        query_time = getattr(query_time, "date", None)
    if not query_time:
        raise ValueError("An explicit public query date is required; input order is not a calendar date")
    return _parse_date(query_time)


def _raw_dates(text):
    """先识别完整日期，再识别月份和年份，避免把同一完整日期降格成别的日期。"""
    ranges = []
    patterns = [re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), _MONTH_FIRST, _DAY_FIRST,
                re.compile(r"\b\d{4}-\d{2}\b"), _MONTH_YEAR, re.compile(r"\b\d{4}\b")]
    for pattern in patterns:
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in ranges):
                continue
            try:
                parsed = _parse_date(match.group())
            except ValueError:
                continue
            ranges.append(match.span())
            yield parsed


def _source_texts(item, bundles, source_text):
    for evidence_id in item.evidence_ids:
        if evidence_id == "@query_time":
            continue
        for ref in bundles[evidence_id].refs:
            if ref.type == "SOURCE":
                for span in [ref.span, *ref.context_refs]:
                    yield source_text(span.source_id)[span.start:span.end]


def _date_operand(key, items, bundles, view, source_text, query_time):
    if key == "@query_time":
        # 这个保留名字只读函数参数，模型伪造同名项目也不能改写公共日期。
        return _public_date(query_time)
    item = items[key]
    chosen = _parse_date(item.value)
    permitted = list(parsed for text in _source_texts(item, bundles, source_text) for parsed in _raw_dates(text))
    for evidence_id in item.evidence_ids:
        if evidence_id == "@query_time":
            permitted.append(_public_date(query_time))
            continue
        for version_id in bundles[evidence_id].version_ids:
            scope = view.versions[version_id].valid_time
            precision = _PRECISION.get(scope.precision)
            if precision is None:
                continue
            for boundary in (scope.start, scope.end):
                if boundary and boundary.date:
                    parsed = _parse_date(boundary.date)
                    permitted.append(_CalendarDate(parsed.year, parsed.month if precision >= 2 else None,
                                                   parsed.day if precision >= 3 else None))
    matches = []
    for evidence in permitted:
        precision = min(_PRECISION[chosen.precision], _PRECISION[evidence.precision])
        if (chosen.year == evidence.year
                and (precision < 2 or chosen.month == evidence.month)
                and (precision < 3 or chosen.day == evidence.day)):
            matches.append(_CalendarDate(chosen.year, chosen.month if precision >= 2 else None,
                                         chosen.day if precision >= 3 else None))
    if not matches:
        raise ValueError("Date operand is not supported by its cited evidence")
    return max(matches, key=lambda item: _PRECISION[item.precision])


def _anniversary(start: date, months: int) -> date:
    index = start.year * 12 + start.month - 1 + months
    year, zero_month = divmod(index, 12)
    month = zero_month + 1
    return date(year, month, min(start.day, monthrange(year, month)[1]))


def _complete_calendar_units(left: date, right: date, months_per_unit: int) -> int:
    start, end = min(left, right), max(left, right)
    units = ((end.year - start.year) * 12 + end.month - start.month) // months_per_unit
    if _anniversary(start, units * months_per_unit) > end:
        units -= 1
    return units if left >= right else -units


def _date_difference(left, right, requested_unit):
    unit = (requested_unit or "days").strip().casefold()
    aliases = {"d": "days", "day": "days", "w": "weeks", "week": "weeks",
               "mo": "months", "month": "months", "y": "years", "yr": "years", "year": "years"}
    unit = aliases.get(unit, unit)
    precision = min(_PRECISION[left.precision], _PRECISION[right.precision])
    extra = {"operand_precision": [left.precision, right.precision]}
    if unit in {"days", "weeks"}:
        days = (left.calendar_day() - right.calendar_day()).days
        value = days if unit == "days" else days / 7
        basis = "exact_calendar_days"
    elif unit in {"months", "years"} and precision == 3:
        value = _complete_calendar_units(left.calendar_day(), right.calendar_day(), 1 if unit == "months" else 12)
        basis = "completed_calendar_months" if unit == "months" else "completed_calendar_years"
        extra["anniversary_rule"] = "original start day, clamped to the last day of a shorter destination month"
    elif unit == "months":
        if precision < 2:
            raise ValueError("Month arithmetic requires evidence at least at month precision")
        value = (left.year - right.year) * 12 + left.month - right.month
        basis = "calendar_month_difference_at_month_precision"
    elif unit == "years":
        if precision == 2:
            months = (left.year - right.year) * 12 + left.month - right.month
            value = (abs(months) // 12) * (-1 if months < 0 else 1)
            basis = "calendar_years_from_month_precision_ignoring_unknown_days"
            extra["calendar_month_difference"] = months
        else:
            value = left.year - right.year
            basis = "calendar_year_difference_at_year_precision"
    else:
        raise ValueError("Date difference supports days, weeks, months, or years")
    return {"value": value, "unit": unit, "basis": basis, **extra}


_UNITS = {}
for names, family, factor, symbol in (
    (("m", "metre", "metres", "meter", "meters"), "m", "1", "m"),
    (("km", "kilometre", "kilometres", "kilometer", "kilometers"), "m", "1000", "km"),
    (("cm", "centimetre", "centimetres", "centimeter", "centimeters"), "m", "0.01", "cm"),
    (("s", "sec", "secs", "second", "seconds"), "seconds", "1", "seconds"),
    (("min", "mins", "minute", "minutes"), "seconds", "60", "minutes"),
    (("h", "hr", "hrs", "hour", "hours"), "seconds", "3600", "hours"),
):
    for name in names:
        _UNITS[name] = (family, Decimal(factor), symbol)


def _unit(value):
    normalized = value.strip().casefold() if value else None
    return _UNITS.get(normalized, (normalized, Decimal(1), normalized))


def compute(calculations, items, bundles, view, source_text, query_time=None, allow_empty_count=False):
    """计算方向保持 item_keys[0] 减 item_keys[1]；日历精度和计算口径写入结果。"""
    by_key = {item.key: item for item in items}
    results = []
    for calculation in calculations:
        keys = calculation.item_keys
        result = {"operation": calculation.operation, "item_keys": keys}
        if calculation.operation == "date_difference":
            if len(keys) != 2:
                raise ValueError("Date difference requires two supported date operands")
            dates = [_date_operand(key, by_key, bundles, view, source_text, query_time) for key in keys]
            results.append({**result, **_date_difference(*dates, calculation.unit)})
            continue
        if "@query_time" in keys:
            raise ValueError("The public query date is only an operand for date_difference")
        selected = [by_key[key] for key in keys]
        if calculation.operation == "count":
            if not selected and not allow_empty_count:
                raise ValueError("An empty count requires exhausted source coverage")
            results.append({**result, "value": len(set(keys)), "unit": None})
            continue
        if not selected:
            raise ValueError("Arithmetic requires source-grounded operands")
        numbers, families = [], []
        for item in selected:
            if item.numeric_value is None:
                raise ValueError("Arithmetic requires explicit numeric operands")
            number = Decimal(item.numeric_value.replace(",", ""))
            literals = {Decimal(match.replace(",", "")) for text in _source_texts(item, bundles, source_text)
                        for match in _NUMBER.findall(text)}
            if not number.is_finite() or number not in literals:
                raise ValueError("Numeric operand does not occur in its cited source")
            family, factor, _ = _unit(item.unit)
            numbers.append(number * factor)
            families.append(family)
        if len(set(families)) != 1:
            raise ValueError("Arithmetic units are incompatible; no currency exchange rates are inferred")
        if calculation.operation == "sum":
            value = sum(numbers, Decimal(0))
        elif len(numbers) != 2:
            raise ValueError("Comparison and difference require two operands")
        elif calculation.operation == "difference":
            value = numbers[0] - numbers[1]
        elif calculation.operation == "compare":
            left, right = numbers
            value = {"<": left < right, "<=": left <= right, "==": left == right,
                     ">=": left >= right, ">": left > right}[calculation.comparator]
            results.append({**result, "value": value, "unit": None})
            continue
        else:
            raise ValueError("Unsupported numeric operation")
        output_unit = families[0]
        if calculation.unit is not None:
            family, factor, symbol = _unit(calculation.unit)
            if family != families[0]:
                raise ValueError("Requested output unit is incompatible with the cited operands")
            value, output_unit = value / factor, symbol
        results.append({**result, "value": str(value), "unit": output_unit})
    return results
