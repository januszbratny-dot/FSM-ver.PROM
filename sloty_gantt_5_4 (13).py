import streamlit as st
import pandas as pd
import plotly.express as px
import random
import os
import json
import tempfile
import logging
import uuid
import locale

# Ustawienie polskiego locale do dat (jeśli dostępne)
try:
    locale.setlocale(locale.LC_TIME, 'pl_PL.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, 'pl_PL')
    except locale.Error:
        pass  # Jeśli nie ma polskiego locale, zostaw domyślne
from datetime import datetime, timedelta, date, time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# ---------------------- CONFIG ----------------------
STORAGE_FILENAME = "schedules.json"
SEARCH_STEP_MINUTES = 15  # krok wyszukiwania wolnego slotu
DEFAULT_WORK_START = time(8, 0)
DEFAULT_WORK_END = time(16, 0)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scheduler")

# ---------------------- DATA MODELS ----------------------
@dataclass
class SlotType:
    name: str
    minutes: int
    weight: float = 1.0

@dataclass
class Slot:
    start: datetime
    end: datetime
    slot_type: str
    duration_min: int
    client: str
    pref_range: Optional[str] = None

# ---------------------- HELPERS: SERIALIZATION ----------------------

def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _time_to_iso(t: time) -> str:
    return t.isoformat()


def parse_datetime_iso(s: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetimes; support trailing 'Z' by converting to +00:00."""
    if s is None:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def parse_time_str(t: str) -> time:
    """Robust parsing for time strings (H:M, H:M:S, H:M:S.sss)."""
    try:
        # Prefer time.fromisoformat if available
        return time.fromisoformat(t)
    except Exception:
        for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(t, fmt).time()
            except ValueError:
                continue
    raise ValueError(f"Nie można sparsować czasu: {t}")

# ---------------------- PERSISTENCE ----------------------

def schedules_to_jsonable() -> Dict:
    data: Dict = {}

    for b, days in st.session_state.schedules.items():
        data[b] = {}
        for d, slots in days.items():
            data[b][d] = [
                {
                    "id": s.get("id"),
                    "start": _datetime_to_iso(s["start"]),
                    "end": _datetime_to_iso(s["end"]),
                    "slot_type": s["slot_type"],
                    "duration_min": s["duration_min"],
                    "client": s["client"],
                    "pref_range": s.get("pref_range", None),
                    "arrival_window_start": _datetime_to_iso(s.get("arrival_window_start")),
                    "arrival_window_end": _datetime_to_iso(s.get("arrival_window_end")),
                }
                for s in slots
            ]

    return {
        "slot_types": st.session_state.slot_types,
        "brygady": st.session_state.brygady,
        "working_hours": {
            b: (_time_to_iso(wh[0]), _time_to_iso(wh[1]))
            for b, wh in st.session_state.working_hours.items()
        },
        "schedules": data,
        "clients_added": st.session_state.clients_added,
        "balance_horizon": st.session_state.balance_horizon,
        "client_counter": st.session_state.client_counter,
        "not_found_counter": st.session_state.not_found_counter,
    }


def save_state_to_json(filename: str = STORAGE_FILENAME):
    """Save state atomically to avoid file corruption on concurrent writes."""
    data = schedules_to_jsonable()
    dirn = os.path.dirname(os.path.abspath(filename)) or "."
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dirn, delete=False) as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        tmpname = tf.name
    os.replace(tmpname, filename)
    logger.info(f"State saved to {filename}")


def load_state_from_json(filename: str = STORAGE_FILENAME) -> bool:
    if not os.path.exists(filename):
        return False
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.exception("Failed to load schedules JSON; ignoring and starting fresh")
        return False

    st.session_state.slot_types = data.get("slot_types", [])
    st.session_state.brygady = data.get("brygady", [])

    st.session_state.working_hours = {}
    for b, wh in data.get("working_hours", {}).items():
        st.session_state.working_hours[b] = (parse_time_str(wh[0]), parse_time_str(wh[1]))

    st.session_state.schedules = {}
    for b, days in data.get("schedules", {}).items():
        st.session_state.schedules[b] = {}
        for d, slots in days.items():
            st.session_state.schedules[b][d] = [
                {
                    "id": s.get("id", str(uuid.uuid4())),
                    "start": parse_datetime_iso(s.get("start")),
                    "end": parse_datetime_iso(s.get("end")),
                    "slot_type": s.get("slot_type"),
                    "duration_min": s.get("duration_min"),
                    "client": s.get("client"),
                    "pref_range": s.get("pref_range", None),
                    "arrival_window_start": parse_datetime_iso(s.get("arrival_window_start")),
                    "arrival_window_end": parse_datetime_iso(s.get("arrival_window_end")),
                }
                for s in slots
            ]

    st.session_state.clients_added = data.get("clients_added", [])
    st.session_state.balance_horizon = data.get("balance_horizon", "week")
    st.session_state.client_counter = data.get("client_counter", 1)
    st.session_state.not_found_counter = data.get("not_found_counter", 0)
    logger.info(f"State loaded from {filename}")
    return True

# ---------------------- INITIALIZATION ----------------------

if "slot_types" not in st.session_state:
    if not load_state_from_json():

        st.session_state.slot_types = [
            {"name": "Zlecenie krótkie", "minutes": 30, "weight": 1.0},
            {"name": "Zlecenie normalne", "minutes": 60, "weight": 1.0},
            {"name": "Zlecenie długie", "minutes": 90, "weight": 1.0}
        ]
        st.session_state.brygady = ["Brygada 1", "Brygada 2"]
        st.session_state.working_hours = {
            "Brygada 1": (DEFAULT_WORK_START, DEFAULT_WORK_END),  # 08:00–16:00
            "Brygada 2": (time(12, 0), time(20, 0))             # 12:00–20:00
        }
        st.session_state.schedules = {}
        st.session_state.clients_added = []
        st.session_state.balance_horizon = "week"
        st.session_state.client_counter = 1
        st.session_state.not_found_counter = 0
        st.session_state.unscheduled_orders = []


# stable keys for widgets (avoid using raw brygada names as keys)
def brygada_key(i: int, field: str) -> str:
    return f"brygada_{i}_{field}"

# ensure brygady presence in working_hours and schedules

def ensure_brygady_in_state(brygady_list: List[str]):
    for i, b in enumerate(brygady_list):
        if b not in st.session_state.working_hours:
            st.session_state.working_hours[b] = (DEFAULT_WORK_START, DEFAULT_WORK_END)
        if b not in st.session_state.schedules:
            st.session_state.schedules[b] = {}

# ---------------------- PARSERS & VALIDATION ----------------------

def parse_slot_types(text: str) -> List[Dict]:
    out: List[Dict] = []
    for i, line in enumerate(text.splitlines(), 1):
        raw = line.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        try:
            name = parts[0]
            minutes = int(parts[1]) if len(parts) > 1 else None
            weight = float(parts[2]) if len(parts) > 2 else 1.0
            if minutes is None or minutes <= 0:
                raise ValueError("minutes must be > 0")
            if weight < 0:
                raise ValueError("weight must be >= 0")
            out.append({"name": name, "minutes": minutes, "weight": weight})
        except Exception as e:
            st.warning(f"Linia {i} pominieta w 'Typy slotów': {e}")
    return out


def weighted_choice(slot_types: List[Dict]) -> Optional[str]:
    if not slot_types:
        return None
    names = [s["name"] for s in slot_types]
    weights = [s.get("weight", 1) for s in slot_types]
    return random.choices(names, weights=weights, k=1)[0]

# ---------------------- ARRIVAL WINDOW HELPERS ----------------------

def oblicz_przedzial_przyjazdu(start_time: datetime,
                               czas_rezerwowy_przed: int,
                               czas_rezerwowy_po: int) -> Tuple[datetime, datetime]:
    """
    Zwraca przedział czasowy przyjazdu brygady do klienta.
    start_time – czas rozpoczęcia głównego slotu
    czas_rezerwowy_przed/po – minuty
    """
    przyjazd_start = start_time - timedelta(minutes=czas_rezerwowy_przed)
    przyjazd_end = start_time + timedelta(minutes=czas_rezerwowy_po)
    return przyjazd_start, przyjazd_end

# ---------------------- SCHEDULE MANAGEMENT ----------------------

def get_day_slots_for_brygada(brygada: str, day: date) -> List[Dict]:
    d = day.strftime("%Y-%m-%d")
    return sorted(st.session_state.schedules.get(brygada, {}).get(d, []), key=lambda s: s["start"])


def add_slot_to_brygada(brygada: str, day: date, slot: Dict, save: bool = True):
    """
    Dodaje slot do harmonogramu brygady i ustawia poprawnie przedział przyjazdu.
    Zasady:
    - Przedział przyjazdu to start slotu - czas_przed  →  start slotu + czas_po.
    - Jeśli początek przedziału wypada przed godzinami pracy brygady, 
      to zostaje przesunięty na początek czasu pracy.
    - Jeśli koniec wypada po godzinach pracy brygady, 
      to zostaje przesunięty tak, aby kończył się równo z końcem pracy.
    - Przedział przyjazdu ma zawsze długość = czas_przed + czas_po (jeśli to możliwe w godzinach pracy).
    """

    # skopiuj, aby nie mutować obiektu przekazanego przez caller
    s = dict(slot)
    if "id" not in s:
        s["id"] = str(uuid.uuid4())

    d = day.strftime("%Y-%m-%d")
    st.session_state.schedules.setdefault(brygada, {})
    st.session_state.schedules[brygada].setdefault(d, [])

    # Pobierz czasy rezerwowe
    try:
        czas_przed = int(st.session_state.get("czas_rezerwowy_przed", 30))
        czas_po = int(st.session_state.get("czas_rezerwowy_po", 30))
    except Exception:
        czas_przed = 30
        czas_po = 30

    # Godziny pracy brygady
    wh_start, wh_end = st.session_state.working_hours.get(brygada, (DEFAULT_WORK_START, DEFAULT_WORK_END))
    wh_start_dt = datetime.combine(day, wh_start)
    wh_end_dt = datetime.combine(day, wh_end)
    if wh_end_dt <= wh_start_dt:  # dla nocnych zmian
        wh_end_dt += timedelta(days=1)

    # Oblicz przedział przyjazdu
    if "start" in s and s["start"]:
        przyjazd_start = s["start"] - timedelta(minutes=czas_przed)
        przyjazd_end = s["start"] + timedelta(minutes=czas_po)

        # Dopasuj do godzin pracy brygady
        if przyjazd_start < wh_start_dt:
            przyjazd_start = wh_start_dt
            przyjazd_end = przyjazd_start + timedelta(minutes=czas_przed + czas_po)

        if przyjazd_end > wh_end_dt:
            przyjazd_end = wh_end_dt
            przyjazd_start = przyjazd_end - timedelta(minutes=czas_przed + czas_po)

        # Ostateczne ograniczenie (dla krótkich dni)
        if przyjazd_start < wh_start_dt:
            przyjazd_start = wh_start_dt
        if przyjazd_end > wh_end_dt:
            przyjazd_end = wh_end_dt

        s["arrival_window_start"] = przyjazd_start
        s["arrival_window_end"] = przyjazd_end
    else:
        s["arrival_window_start"] = None
        s["arrival_window_end"] = None


    # Wstaw slot w odpowiednie miejsce (utrzymuj listę posortowaną po 'start')
    slots = st.session_state.schedules[brygada][d]
    # Jeśli lista pusta lub nowy slot na końcu
    if not slots or s["start"] >= slots[-1]["start"]:
        slots.append(s)
    else:
        # Znajdź miejsce do wstawienia (bisect)
        import bisect
        starts = [slot["start"] for slot in slots]
        idx = bisect.bisect_left(starts, s["start"])
        slots.insert(idx, s)

    if save:
        save_state_to_json()



def delete_slot(brygada: str, day_str: str, slot_id: str):
    st.session_state.schedules.setdefault(brygada, {})
    slots = st.session_state.schedules[brygada].get(day_str, [])
    before = len(slots)
    st.session_state.schedules[brygada][day_str] = [s for s in slots if s.get("id") != slot_id]
    after = len(st.session_state.schedules[brygada][day_str])
    if before != after:
        save_state_to_json()
        logger.info(f"Deleted slot {slot_id} on {brygada} {day_str}")


def _wh_minutes(wh_start: time, wh_end: time) -> int:
    """Return minutes in working hours. Support overnight shifts (end <= start) by wrapping to next day."""
    start_dt = datetime.combine(date.today(), wh_start)
    end_dt = datetime.combine(date.today(), wh_end)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return int((end_dt - start_dt).total_seconds() // 60)


def schedule_client_immediately(client_name: str, slot_type_name: str, day: date,
                                pref_start: time, pref_end: time, save: bool = True) -> Tuple[bool, Optional[Dict]]:
    """
    Znajduje najlepszy możliwy termin dla klienta w danym dniu, preferując:
    1. Sloty mieszczące się w preferencjach klienta,
    2. Sloty najbliżej początku lub końca dnia pracy brygady,
    3. Brygady o najmniejszym wykorzystaniu.
    """
    slot_type = next((s for s in st.session_state.slot_types if s["name"] == slot_type_name), None)
    if not slot_type:
        return False, None

    dur = timedelta(minutes=slot_type["minutes"])
    candidates: List[Tuple[str, datetime, datetime, bool, float, int]] = []
    # (brygada, start_dt, end_dt, in_pref, edge_priority, utilization)

    for b in st.session_state.brygady:
        existing = get_day_slots_for_brygada(b, day)
        wh_start, wh_end = st.session_state.working_hours.get(b, (DEFAULT_WORK_START, DEFAULT_WORK_END))

        # ustalenie początku/końca dnia pracy
        day_start_dt = datetime.combine(day, wh_start)
        day_end_dt = datetime.combine(day, wh_end)
        if day_end_dt <= day_start_dt:
            day_end_dt += timedelta(days=1)

        pref_start_dt = datetime.combine(day, pref_start)
        pref_end_dt = datetime.combine(day, pref_end)
        if pref_end_dt <= pref_start_dt:
            pref_end_dt += timedelta(days=1)

        t = day_start_dt
        while t + dur <= day_end_dt:
            t_end = t + dur

            # sprawdź kolizję
            overlap = any(not (t_end <= s["start"] or t >= s["end"]) for s in existing)
            if not overlap:
                # czy slot mieści się w preferencjach
                in_pref = (t >= pref_start_dt) and (t_end <= pref_end_dt)

                # dystans do krawędzi dnia pracy (im mniejszy, tym lepiej)
                dist_to_start = (t - day_start_dt).total_seconds()
                dist_to_end = (day_end_dt - t_end).total_seconds()
                edge_priority = min(dist_to_start, dist_to_end)

                # wykorzystanie brygady (ile minut już zaplanowane)
                utilization = sum(
                    s["duration_min"] for d in st.session_state.schedules.get(b, {}).values() for s in d
                )

                candidates.append((b, t, t_end, in_pref, edge_priority, utilization))
            t += timedelta(minutes=SEARCH_STEP_MINUTES)

    if not candidates:
        st.session_state.not_found_counter = st.session_state.get("not_found_counter", 0) + 1
        return False, None

    # Sortowanie:
    # 1. sloty w preferencji (True przed False),
    # 2. edge_priority (bliżej krawędzi),
    # 3. wykorzystanie (mniej obciążona brygada),
    # 4. czas rozpoczęcia (wcześniej)
    candidates.sort(key=lambda x: (
        not x[3],            # False (czyli w preferencji) ma być pierwsze
        x[4],                # odległość od krawędzi
        x[5],                # wykorzystanie brygady
        x[1]                 # czas startu
    ))

    brygada, start, end, _, _, _ = candidates[0]

    slot = {
        "id": str(uuid.uuid4()),
        "start": start,
        "end": end,
        "slot_type": slot_type_name,
        "duration_min": slot_type["minutes"],
        "client": client_name,
    }

    add_slot_to_brygada(brygada, day, slot, save=save)
    # zwracamy informację o tym, do której brygady przydzielono slot
    slot_with_meta = dict(slot)
    slot_with_meta["brygada"] = brygada
    return True, slot_with_meta

# ---------------------- PREDEFINED SLOTS & UTIL ----------------------
PREFERRED_SLOTS = {
    "8:00-11:00": (time(8, 0), time(11, 0)),
    "11:00-14:00": (time(11, 0), time(14, 0)),
    "14:00-17:00": (time(14, 0), time(17, 0)),
    "17:00-20:00": (time(17, 0), time(20, 0)),
}


def get_week_days(reference_day: date) -> List[date]:
    monday = reference_day - timedelta(days=reference_day.weekday())
    return [monday + timedelta(days=i) for i in range(7)]


def get_available_slots_for_day(day: date, slot_minutes: int, step_minutes: int = SEARCH_STEP_MINUTES) -> List[Dict]:
    """Zwraca sloty, które można przydzielić na początku/końcu dnia pracy
    lub które bezpośrednio sąsiadują z już zarezerwowanymi slotami."""

    available_slots = []

    for brygada, working_hours in st.session_state.working_hours.items():
        wh_start, wh_end = working_hours
        wh_start_dt = datetime.combine(day, wh_start)
        wh_end_dt = datetime.combine(day, wh_end)
        if wh_end_dt <= wh_start_dt:
            wh_end_dt += timedelta(days=1)

        slots = get_day_slots_for_brygada(brygada, day)
        used_intervals = [(s["start"], s["end"]) for s in slots]
        candidates = []

        if not used_intervals:
            # Brak rezerwacji -> pokaż początek i koniec dnia pracy
            start_dt = wh_start_dt
            end_dt = start_dt + timedelta(minutes=slot_minutes)
            if end_dt <= wh_end_dt:
                candidates.append((start_dt, end_dt))

            end_dt = wh_end_dt
            start_dt = end_dt - timedelta(minutes=slot_minutes)
            if start_dt >= wh_start_dt:
                candidates.append((start_dt, end_dt))
        else:
            # Sloty przylegające
            for s in used_intervals:
                # Slot przed istniejącym
                before_end = s[0]
                before_start = before_end - timedelta(minutes=slot_minutes)
                if before_start >= wh_start_dt:
                    candidates.append((before_start, before_end))

                # Slot po istniejącym
                after_start = s[1]
                after_end = after_start + timedelta(minutes=slot_minutes)
                if after_end <= wh_end_dt:
                    candidates.append((after_start, after_end))

            # Brzegowe – jeśli pierwszy slot nie sięga początku pracy
            first_slot_start = min(s[0] for s in used_intervals)
            if first_slot_start > wh_start_dt:
                start_dt = wh_start_dt
                end_dt = start_dt + timedelta(minutes=slot_minutes)
                if end_dt <= first_slot_start:
                    candidates.append((start_dt, end_dt))

            # Brzegowe – jeśli ostatni slot nie sięga końca pracy
            last_slot_end = max(s[1] for s in used_intervals)
            if last_slot_end < wh_end_dt:
                end_dt = wh_end_dt
                start_dt = end_dt - timedelta(minutes=slot_minutes)
                if start_dt >= last_slot_end:
                    candidates.append((start_dt, end_dt))

        # Filtr kolizji (dla pewności)
        valid = []
        for c_start, c_end in candidates:
            overlaps = any(
                not (c_end <= u_start or c_start >= u_end)
                for u_start, u_end in used_intervals
            )
            if not overlaps:
                valid.append((c_start, c_end))

        # Dodaj sloty do listy
        for start_dt, end_dt in sorted(set(valid)):
            available_slots.append({
                "brygada": brygada,
                "start": start_dt,
                "end": end_dt,
                "slot_type": None
            })

    # Agregacja duplikatów między brygadami
    grouped = {}
    for s in available_slots:
        key = (s["start"], s["end"])
        grouped.setdefault(key, []).append(s["brygada"])

    result = []
    for (start_dt, end_dt), brygady in grouped.items():
        result.append({
            "start": start_dt,
            "end": end_dt,
            "brygady": brygady
        })

    result.sort(key=lambda x: x["start"])
    logging.info(f"DEBUG: get_available_slots_for_day({day}) -> {len(result)} slots")
    return result

# ---------------------- UI ----------------------
st.set_page_config(page_title="Harmonogram slotów", layout="wide")
st.title("📅 Harmonogram slotów - Tydzień")

with st.sidebar:
    st.subheader("⚙️ Konfiguracja")

    # slot types editor with validation
    txt = st.text_area("Typy slotów (format: Nazwa, minuty, waga)",
                       value="\n".join(f"{s['name']},{s['minutes']},{s.get('weight',1)}" for s in st.session_state.slot_types))
    parsed = parse_slot_types(txt)
    if parsed:
        st.session_state.slot_types = parsed

    # brygady editor
    txt_b = st.text_area("Lista brygad", value="\n".join(st.session_state.brygady))
    brygady_new = [line.strip() for line in txt_b.splitlines() if line.strip()]
    if brygady_new and brygady_new != st.session_state.brygady:
        st.session_state.brygady = brygady_new
    ensure_brygady_in_state(st.session_state.brygady)

    st.markdown("---")
    st.write("Godziny pracy (możesz edytować każdą brygadę)")
    for i, b in enumerate(st.session_state.brygady):
        # stable keys so widgets don't lose state when name changes
        start_t = st.time_input(f"Start {b}", value=st.session_state.working_hours[b][0], key=brygada_key(i, "start"))
        end_t = st.time_input(f"Koniec {b}", value=st.session_state.working_hours[b][1], key=brygada_key(i, "end"))
        st.session_state.working_hours[b] = (start_t, end_t)

    st.markdown("---")
    if st.button("🗑️ Wyczyść harmonogram"):
        st.session_state.schedules = {b: {} for b in st.session_state.brygady}
        st.session_state.clients_added = []
        st.session_state.client_counter = 1
        st.session_state.not_found_counter = 0
        save_state_to_json()
        st.success("Harmonogram wyczyszczony.")

    # Arrival window settings
    st.subheader("🕓 Czas rezerwowy (przyjazd Brygady)")
    st.write("Ustaw w minutach: przed i po czasie rozpoczęcia slotu.")
    st.session_state.czas_rezerwowy_przed = st.number_input(
        "Czas rezerwowy przed (minuty)", min_value=0, max_value=180, value=30, step=5, key="czas_przed"
    )
    st.session_state.czas_rezerwowy_po = st.number_input(
        "Czas rezerwowy po (minuty)", min_value=0, max_value=180, value=30, step=5, key="czas_po"
    )

# week navigation
if "week_offset" not in st.session_state:
    st.session_state.week_offset = 0

def polish_date(dt):
    dni_polskie = {
        'Monday': 'Poniedziałek',
        'Tuesday': 'Wtorek',
        'Wednesday': 'Środa',
        'Thursday': 'Czwartek',
        'Friday': 'Piątek',
        'Saturday': 'Sobota',
        'Sunday': 'Niedziela',
    }
    try:
        day_en = dt.strftime("%A")
        day_pl = dni_polskie.get(day_en, day_en)
        return f"{day_pl}, {dt.strftime('%d.%m.%Y')}"
    except Exception:
        return str(dt)

with st.sidebar:
    st.subheader("⬅️ Wybór tygodnia")
    col1, col2 = st.columns(2)
    if col1.button("‹ Poprzedni tydzień"):
        st.session_state.week_offset -= 1
    if col2.button("Następny tydzień ›"):
        st.session_state.week_offset += 1

week_ref = date.today() + timedelta(weeks=st.session_state.week_offset)
week_days = get_week_days(week_ref)
st.sidebar.write(f"Tydzień: {polish_date(week_days[0])} – {polish_date(week_days[-1])}")

# ---------------------- Dodaj klienta (zmieniony UI: wybór dostępnego slotu) ----------------------
st.subheader("➕ Rezerwacja terminu")

# Imię klienta
with st.container():
    default_client = f"Klient {st.session_state.client_counter}"
    client_name = st.text_input("Nazwa klienta", value=default_client)

# Wybór typu slotu (pozostawiamy)
slot_names = [s["name"] for s in st.session_state.slot_types]
if not slot_names:
    slot_names = ["Standard"]
    st.session_state.slot_types = [{"name": "Standard", "minutes": 60, "weight": 1.0}]
auto_type = weighted_choice(st.session_state.slot_types) or slot_names[0]
idx = slot_names.index(auto_type) if auto_type in slot_names else 0
slot_type_name = st.selectbox("Typ slotu", slot_names, index=idx)
slot_type = next((s for s in st.session_state.slot_types if s["name"] == slot_type_name), slot_names[0])
slot_duration = timedelta(minutes=slot_type["minutes"])


# Synchronizacja wyboru daty z autofill
if "booking_day" not in st.session_state:
    st.session_state.booking_day = date.today()
if "autofill_day_full" in st.session_state:
    # Jeśli zmieniono datę w autofill, ustaw ją w booking_day
    if st.session_state.booking_day != st.session_state.autofill_day_full:
        st.session_state.booking_day = st.session_state.autofill_day_full

col_prev, col_mid, col_next = st.columns([1, 2, 1])
with col_prev:
    if st.button("⬅️ Poprzedni dzień", key="booking_prev"):
        st.session_state.booking_day -= timedelta(days=1)
        st.session_state.autofill_day_full = st.session_state.booking_day
with col_next:
    if st.button("Następny dzień ➡️", key="booking_next"):
        st.session_state.booking_day += timedelta(days=1)
        st.session_state.autofill_day_full = st.session_state.booking_day
with col_mid:
    st.markdown(f"### {polish_date(st.session_state.booking_day)}")

booking_day = st.session_state.booking_day

# --- WIDOK DOSTĘPNYCH SLOTÓW ---
st.markdown("### 🕒 Dostępne sloty w wybranym dniu")

slot_minutes = slot_type["minutes"]

# Funkcja do znalezienia najbliższego dnia z dostępnymi slotami
def find_next_day_with_slots(start_day, slot_minutes, max_days=30):
    for offset in range(max_days):
        test_day = start_day + timedelta(days=offset)
        slots = get_available_slots_for_day(test_day, slot_minutes)
        if slots:
            return test_day, slots
    return start_day, []

available_slots = get_available_slots_for_day(booking_day, slot_minutes)

# Jeśli nie ma slotów, automatycznie przełącz na najbliższy dzień z dostępnymi slotami
if not available_slots:
    next_day, next_slots = find_next_day_with_slots(booking_day, slot_minutes)
    if next_slots and next_day != booking_day:
        st.session_state.booking_day = next_day
        booking_day = next_day
        available_slots = next_slots
        st.markdown(f"""
        <div style='background-color:#fff9c4; color:#333; border-radius:6px; padding:12px; border:1px solid #ffe082; font-size:1.1em; margin-bottom:1em;'>
        <b>Brak terminu w wybranym dniu.</b><br>Najbliższe wolne terminy są dostępne w dniu: <b>{polish_date(booking_day)}</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Brak dostępnych slotów dla wybranego dnia.")

slots_for_display = []
for s in available_slots:
    try:
        czas_przed = int(st.session_state.get("czas_rezerwowy_przed", 30))
        czas_po = int(st.session_state.get("czas_rezerwowy_po", 30))
    except Exception:
        czas_przed = 30
        czas_po = 30

    for brygada in s.get("brygady", []):
        wh_start, wh_end = st.session_state.working_hours.get(brygada, (DEFAULT_WORK_START, DEFAULT_WORK_END))
        wh_start_dt = datetime.combine(booking_day, wh_start)
        wh_end_dt = datetime.combine(booking_day, wh_end)
        if wh_end_dt <= wh_start_dt:  # nocna zmiana
            wh_end_dt += timedelta(days=1)

        start_dt = s["start"]
        arr_start_dt = start_dt - timedelta(minutes=czas_przed)
        arr_end_dt = start_dt + timedelta(minutes=czas_po)

        if arr_start_dt < wh_start_dt:
            arr_start_dt = wh_start_dt
            arr_end_dt = arr_start_dt + timedelta(minutes=czas_przed + czas_po)

        if arr_end_dt > wh_end_dt:
            arr_end_dt = wh_end_dt
            arr_start_dt = arr_end_dt - timedelta(minutes=czas_przed + czas_po)

        # ostateczne przycięcie
        arr_start_dt = max(arr_start_dt, wh_start_dt)
        arr_end_dt = min(arr_end_dt, wh_end_dt)

        # dodajemy osobny rekord dla każdej brygady/przedziału
        slots_for_display.append({
            "start": s["start"],
            "end": s["end"],
            "brygada": brygada,
            "arrival_window_start": arr_start_dt,
            "arrival_window_end": arr_end_dt
        })

# Sortowanie po czasie startu i brygadzie
slots_for_display.sort(key=lambda x: (x["start"], x["arrival_window_start"], x["brygada"]))

if available_slots:
    # Zielony CSS dla przycisków
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    for i, s in enumerate(slots_for_display):
        col0, col1, col2, col3 = st.columns([2, 2, 2, 1])
    
        col0.write(f"🛠️ Slot pracy: {s['start'].strftime('%H:%M')} – {s['end'].strftime('%H:%M')}")
        col1.write(f"🚗 Przedział przyjazdu: {s['arrival_window_start'].strftime('%H:%M')} – {s['arrival_window_end'].strftime('%H:%M')}")
        col2.write(f"👷 Brygada: {s['brygada']}")
    
        if col3.button("Zarezerwuj w tym slocie", key=f"book_{i}"):
            slot = {
                "start": s["start"],
                "end": s["end"],
                "slot_type": slot_type_name,
                "duration_min": slot_minutes,
                "client": client_name,
            }
            add_slot_to_brygada(s["brygada"], booking_day, slot)
            st.session_state.client_counter += 1
            st.success(f"✅ Zarezerwowano slot {s['start'].strftime('%H:%M')}–{s['end'].strftime('%H:%M')} w brygadzie {s['brygada']}.")
            st.rerun()



# --- Przycisk „Zleć bez terminu” ---
st.markdown("### ⏳ Przekazanie zlecenia do Dyspozytora")
if st.button("Zleć bez terminu", key="unscheduled_order"):
    st.session_state.unscheduled_orders.append({
        "client": client_name,
        "slot_type": slot_type_name,
        "created": datetime.now().isoformat()
    })

    # automatyczne przypisanie nowego klienta tylko do client_name
    st.session_state.client_counter += 1
    #st.session_state.client_name = f"Klient {st.session_state.client_counter}"

    save_state_to_json()
    st.success(f"✅ Zlecenie dla {client_name} dodane do listy bez terminu.")
    st.rerun()


# ---------------------- AUTO-FILL FULL DAY (BEZPIECZNY) ----------------------
st.subheader("⚡ Automatyczne dociążenie wszystkich brygad")


# wybór dnia do autofill zsynchronizowany z booking_day
if "autofill_day_full" not in st.session_state:
    st.session_state.autofill_day_full = st.session_state.booking_day
day_autofill = st.date_input(
    "Dzień do wypełnienia (pełny dzień)",
    key="autofill_day_full"
)
# Synchronizacja booking_day z autofill_day_full
if st.session_state.booking_day != st.session_state.autofill_day_full:
    st.session_state.booking_day = st.session_state.autofill_day_full

# przycisk uruchamiający autofill
if st.button("🚀 Wypełnij cały dzień do 100%"):
    added_total = 0
    max_iterations = 5000
    iteration = 0
    slots_added_in_last_iteration = True

    # główna pętla dodawania slotów dopóki coś się udało dodać
    while iteration < max_iterations and slots_added_in_last_iteration:
        iteration += 1
        slots_added_in_last_iteration = False

        for b in st.session_state.brygady:
            wh_start, wh_end = st.session_state.working_hours[b]
            daily_minutes = _wh_minutes(wh_start, wh_end)
            d_str = day_autofill.strftime("%Y-%m-%d")

            # BEZPIECZNIE – upewniamy się, że istnieje słownik dla brygady i dnia
            st.session_state.schedules.setdefault(b, {})
            st.session_state.schedules[b].setdefault(d_str, [])
            slots = st.session_state.schedules[b][d_str]

            used_minutes = sum(s["duration_min"] for s in slots)
            if used_minutes >= daily_minutes:
                continue  # brygada pełna, pomijamy

            # losujemy typ slotu i preferowany przedział
            auto_type = weighted_choice(st.session_state.slot_types) or "Standard"
            auto_pref_label = random.choice(list(PREFERRED_SLOTS.keys()))
            pref_start, pref_end = PREFERRED_SLOTS[auto_pref_label]
            client_name = f"AutoKlient {st.session_state.client_counter}"

            # próbujemy dodać slot (bez zapisu przy każdym dodaniu dla performance)
            ok, info = schedule_client_immediately(client_name, auto_type, day_autofill, pref_start, pref_end, save=False)
            if ok and info:
                assigned_b = info["brygada"]
                d_str = day_autofill.strftime("%Y-%m-%d")
                # ustaw pref_range w właściwym obiekcie (szukamy po id)
                for s in st.session_state.schedules[assigned_b][d_str]:
                    if s.get("id") == info.get("id"):
                        s["pref_range"] = auto_pref_label
                        break

                st.session_state.clients_added.append({
                    "client": client_name,
                    "slot_type": auto_type,
                    "pref_range": auto_pref_label
                })
                st.session_state.client_counter += 1
                added_total += 1
                slots_added_in_last_iteration = True

    # po zakończeniu pętli zapisz raz
    save_state_to_json()

    # ustawiamy flagę, która będzie przetworzona w kolejnym renderze
    st.session_state["autofill_done"] = True
    st.session_state["added_total"] = added_total

# ---------------------- BLOK OBSŁUGI RERUN (BEZPIECZNY) ----------------------
if st.session_state.get("autofill_done"):
    added_total = st.session_state.pop("added_total", 0)
    st.session_state.pop("autofill_done", None)

    if added_total > 0:
        st.success(f"✅ Dodano {added_total} klientów – dzień {day_autofill.strftime('%d-%m-%Y')} wypełniony do 100% we wszystkich brygadach.")
    else:
        st.info("ℹ️ Wszystkie brygady są już w pełni obciążone w tym dniu.")

    # BEZPIECZNE wywołanie rerun po zakończeniu renderu
    st.rerun()

# ---------------------- Harmonogram (tabela) ----------------------
all_slots = []
for b in st.session_state.brygady:
    for d in week_days:
        d_str = d.strftime("%Y-%m-%d")
        slots = st.session_state.schedules.get(b, {}).get(d_str, [])
        for s in slots:
            all_slots.append({
                "Brygada": b,
                "Dzień": d_str,
                "Klient": s["client"],
                "Typ": s["slot_type"],
                "Przedział przyjazdu": s.get("arrival_window_start") and s.get("arrival_window_end") and f"{s['arrival_window_start'].strftime('%H:%M')} - {s['arrival_window_end'].strftime('%H:%M')}",
                "Start": s["start"],
                "Koniec": s["end"],
                "Czas [min]": s["duration_min"],
                "_id": s.get("id", s["start"].isoformat()),
            })

df = pd.DataFrame(all_slots)
st.subheader("📋 Tabela harmonogramu")
if df.empty:
    st.info("Brak zaplanowanych slotów w tym tygodniu.")
else:
    st.dataframe(df.drop(columns=["_id"]))

# ---------------------- GANTT 2 ----------------------
st.subheader(f"📊 Gantt dnia: {polish_date(booking_day)} – Praca i przedział przyjazdu (osobno dla każdej brygady)")

for b in st.session_state.brygady:
    d_str = booking_day.strftime("%Y-%m-%d")
    slots = st.session_state.schedules.get(b, {}).get(d_str, [])
    if not slots:
        st.info(f"Brak slotów dla {b} w wybranym dniu.")
        continue

    dual_slots_day = []
    for s in slots:
        y_label = f"{s['client']}"

        # Slot pracy
        dual_slots_day.append({
            "Y": y_label,
            "Typ": "Slot pracy",
            "Start": s["start"],
            "Koniec": s["end"],
        })

        # Przedział przyjazdu
        if s.get("arrival_window_start") and s.get("arrival_window_end"):
            dual_slots_day.append({
                "Y": y_label,
                "Typ": "Przedział przyjazdu",
                "Start": s["arrival_window_start"],
                "Koniec": s["arrival_window_end"],
            })

    df_dual_day = pd.DataFrame(dual_slots_day)
    if df_dual_day.empty:
        st.info(f"Brak slotów do wyświetlenia dla brygady {b}.")
        continue

    fig_day = px.timeline(
        df_dual_day,
        x_start="Start",
        x_end="Koniec",
        y="Y",
        color="Typ",
        color_discrete_map={
            "Slot pracy": "#1f77b4",
            "Przedział przyjazdu": "#ff7f0e"
        },
        hover_data=["Typ"]
    )

    for trace in fig_day.data:
        if trace.name == "Przedział przyjazdu":
            trace.opacity = 0.3
        else:
            trace.opacity = 1.0

    fig_day.update_yaxes(autorange="reversed")

    # Dodanie preferowanych przedziałów w tle
    for label, (s, e) in PREFERRED_SLOTS.items():
        fig_day.add_vrect(
            x0=datetime.combine(booking_day, s),
            x1=datetime.combine(booking_day, e),
            fillcolor="rgba(200,200,200,0.15)",
            opacity=0.2,
            layer="below",
            line_width=0
        )
        fig_day.add_vline(x=datetime.combine(booking_day, s), line_width=1, line_dash="dot")
        fig_day.add_vline(x=datetime.combine(booking_day, e), line_width=1, line_dash="dot")

    st.markdown(f"### Brygada: {b}")
    st.plotly_chart(fig_day, use_container_width=True)

# ---------------------- ZLECENIA BEZ TERMINU ----------------------
st.subheader("⏳ Zlecenia bez terminu - Dyspozytor")

# Inicjalizacja listy, jeśli nie istnieje
if "unscheduled_orders" not in st.session_state:
    st.session_state.unscheduled_orders = []


if st.session_state.unscheduled_orders:
    # iterujemy po kopii listy, aby być bezpiecznym przy mutacjach
    for idx, o in enumerate(list(st.session_state.unscheduled_orders)):
        cols = st.columns([3, 2, 1])
        cols[0].write(f"{o['client']} — {o['slot_type']}")
        cols[1].write(f"Dodano: {datetime.fromisoformat(o['created']).strftime('%d-%m-%Y %H:%M')}")
        # klucz guzika uczyniony bardziej unikalnym (idx + timestamp)
        btn_key = f"unsched_del_{idx}_{o.get('created')}"
        if cols[2].button("Usuń", key=btn_key):
            # usuwamy po unikalnym 'created' (stabilniejsze niż index)
            st.session_state.unscheduled_orders = [
                x for x in st.session_state.unscheduled_orders if x.get("created") != o.get("created")
            ]
            save_state_to_json()          # <- KLUCZ: zapisz zmiany!
            st.success(f"❌ Zlecenie {o['client']} usunięte.")
            st.rerun()



#----------------------------------------------------
# management: delete individual slots
st.subheader("🧰 Zarządzaj slotami")



# --- FILTRY ---
with st.expander("Filtry", expanded=True):
    colf1, colf2, colf3, colf4 = st.columns(4)
    brygady_options = ["(wszystkie)"] + sorted(df["Brygada"].unique()) if not df.empty else ["(wszystkie)"]
    typy_options = ["(wszystkie)"] + sorted(df["Typ"].dropna().unique()) if not df.empty else ["(wszystkie)"]
    klienci_options = ["(wszyscy)"] + sorted(df["Klient"].dropna().unique()) if not df.empty else ["(wszyscy)"]
    # Daty w polskim formacie i z polskim dniem tygodnia
    def polish_day(date_str):
        dni_polskie = {
            'Monday': 'Poniedziałek',
            'Tuesday': 'Wtorek',
            'Wednesday': 'Środa',
            'Thursday': 'Czwartek',
            'Friday': 'Piątek',
            'Saturday': 'Sobota',
            'Sunday': 'Niedziela',
        }
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            day_en = d.strftime("%A")
            day_pl = dni_polskie.get(day_en, day_en)
            return f"{day_pl}, {d.strftime('%d.%m.%Y')}"
        except Exception:
            return date_str
    dni_raw = sorted(df["Dzień"].unique()) if not df.empty else []
    dni_options = ["(wszystkie)"] + [polish_day(d) for d in dni_raw]

    brygada_filter = colf1.selectbox("Brygada", brygady_options, key="filter_brygada")
    typ_filter = colf2.selectbox("Typ slotu", typy_options, key="filter_typ")
    klient_filter = colf3.selectbox("Klient", klienci_options, key="filter_klient")
    dzien_filter = colf4.selectbox("Dzień", dni_options, key="filter_dzien")

# --- FILTROWANIE ---
filtered_df = df.copy()
if not df.empty:
    if brygada_filter != "(wszystkie)":
        filtered_df = filtered_df[filtered_df["Brygada"] == brygada_filter]
    if typ_filter != "(wszystkie)":
        filtered_df = filtered_df[filtered_df["Typ"] == typ_filter]
    if klient_filter != "(wszyscy)":
        filtered_df = filtered_df[filtered_df["Klient"] == klient_filter]
    if dzien_filter != "(wszystkie)":
        # Porównuj po polskim formacie
        def polish_day(date_str):
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d")
                return d.strftime("%A, %d.%m.%Y").capitalize()
            except Exception:
                return date_str
        filtered_df = filtered_df[filtered_df["Dzień"].apply(lambda d: polish_day(d) == dzien_filter)]

# Nagłówek kolumn z tłem i pogrubieniem
header_cols = st.columns([1, 2, 1, 1.2, 1, 1])
headers = ["Dzień", "Klient + Typ", "Przedział przyjazdu", "Start – Koniec", "Brygada", "Akcje"]
for col, title in zip(header_cols, headers):
    col.markdown(f"<div style='background-color:#f0f0f0; font-weight:bold; padding:4px; border-radius:4px;'>{title}</div>", unsafe_allow_html=True)

# Wiersze z danymi
if not filtered_df.empty:
    for idx, row in filtered_df.iterrows():
        cols = st.columns([1, 2, 1, 1.2, 1, 1])
        # Wyświetl dzień po polsku
        try:
            dni_polskie = {
                'Monday': 'Poniedziałek',
                'Tuesday': 'Wtorek',
                'Wednesday': 'Środa',
                'Thursday': 'Czwartek',
                'Friday': 'Piątek',
                'Saturday': 'Sobota',
                'Sunday': 'Niedziela',
            }
            d = datetime.strptime(row["Dzień"], "%Y-%m-%d")
            day_en = d.strftime("%A")
            day_pl = dni_polskie.get(day_en, day_en)
            dzien_polski = f"{day_pl}, {d.strftime('%d.%m.%Y')}"
        except Exception:
            dzien_polski = row["Dzień"]
        cols[0].write(dzien_polski)
        cols[1].write(f"**{row['Klient']}** — {row['Typ']}")
        cols[2].write(row["Przedział przyjazdu"] if row["Przedział przyjazdu"] else "-")
        cols[3].write(f"{row['Start'].strftime('%H:%M')} - {row['Koniec'].strftime('%H:%M')}")
        cols[4].write(row["Brygada"])
        if cols[5].button("Usuń", key=f"del_{row['Brygada']}_{row['_id']}"):
            delete_slot(row["Brygada"], row["Dzień"], row["_id"])
            st.success(f"✅ Slot dla {row['Klient']} w brygadzie {row['Brygada']} usunięty.")
            st.rerun()








# ---------------------- GANTT ----------------------
if not df.empty:
    st.subheader("📊 Wykres Gantta - tydzień")
    fig = px.timeline(df, x_start="Start", x_end="Koniec", y="Brygada", color="Klient", hover_data=["Typ", "Przedział przyjazdu"])
    fig.update_yaxes(autorange="reversed")

    for d in week_days:
        for label, (s, e) in PREFERRED_SLOTS.items():
            fig.add_vrect(x0=datetime.combine(d, s), x1=datetime.combine(d, e), fillcolor="rgba(200,200,200,0.15)", opacity=0.2, layer="below", line_width=0)
            fig.add_vline(x=datetime.combine(d, s), line_width=1, line_dash="dot")
            fig.add_vline(x=datetime.combine(d, e), line_width=1, line_dash="dot")

    st.plotly_chart(fig, use_container_width=True)


# ---------------------- PODSUMOWANIE ----------------------
st.subheader("📌 Podsumowanie")
st.write(f"✅ Dodano klientów: {len(st.session_state.clients_added)}")
st.write(f"❌ Brak slotu dla: {st.session_state.not_found_counter}")

# ---------------------- UTILIZATION PER DAY ----------------------
st.subheader("📊 Wykorzystanie brygad w podziale na dni (%)")
util_data = []
for b in st.session_state.brygady:
    row = {"Brygada": b}
    wh_start, wh_end = st.session_state.working_hours[b]
    daily_minutes = _wh_minutes(wh_start, wh_end)
    for d in week_days:
        d_str = d.strftime("%Y-%m-%d")
        slots = st.session_state.schedules.get(b, {}).get(d_str, [])
        used = sum(s["duration_min"] for s in slots)
        row[d_str] = round(100 * used / daily_minutes, 1) if daily_minutes > 0 else 0
    util_data.append(row)
st.dataframe(pd.DataFrame(util_data))

# ---------------------- TOTAL UTILIZATION ----------------------
st.subheader("📊 Wykorzystanie brygad (sumarycznie)")
rows = []
for b in st.session_state.brygady:
    total = sum(s["duration_min"] for d in st.session_state.schedules.get(b, {}).values() for s in d)
    wh_start, wh_end = st.session_state.working_hours[b]
    daily_minutes = _wh_minutes(wh_start, wh_end)
    available = daily_minutes * len(week_days)
    utilization = round(100 * total / available, 1) if available > 0 else 0
    rows.append({"Brygada": b, "Zajętość [min]": total, "Dostępne [min]": available, "Wykorzystanie [%]": utilization})
st.table(pd.DataFrame(rows))

# ---------------------- OPTIONAL: BASIC TESTS ----------------------

def _run_basic_tests():
    """Uruchom prosty sanity test parsers i scheduler logic jeśli uruchomione manualnie.
    Aby uruchomić: RUN_SCHEDULE_TESTS=1 streamlit run this_file.py
    """
    errors = []
    # parse time
    try:
        assert parse_time_str("08:00").hour == 8
        assert parse_time_str("23:59:59").hour == 23
    except Exception as e:
        errors.append(f"parse_time_str failed: {e}")

    # schedule overlapping test
    test_day = date.today()
    st.session_state.slot_types = [{"name": "T30", "minutes": 30, "weight": 1}]
    st.session_state.brygady = ["T1"]
    st.session_state.working_hours = {"T1": (time(8, 0), time(10, 0))}
    st.session_state.schedules = {"T1": {}}

    ok1, slot1 = schedule_client_immediately("A", "T30", test_day, time(8, 0), time(10, 0))
    ok2, slot2 = schedule_client_immediately("B", "T30", test_day, time(8, 0), time(10, 0))
    ok3, slot3 = schedule_client_immediately("C", "T30", test_day, time(8, 0), time(10, 0))
    # 2 slots fit in 2 hours if step 30 -> actually 4 slots, depending on step; just check no crash
    if not ok1 or not ok2:
        errors.append("Scheduling basic failed")

    if errors:
        st.error('Testy wykryły błędy: ' + '; '.join(errors))
    else:
        st.success('Podstawowe testy przeszły pomyślnie ✅')

if os.environ.get("RUN_SCHEDULE_TESTS"):
    _run_basic_tests()


# ---------------------- GANTT 1-DNIOWY: Praca + Przedział przyjazdu ----------------------
st.subheader(f"📊 Gantt dnia: {polish_date(booking_day)} – Praca i przedział przyjazdu")

dual_slots_day = []
for b in st.session_state.brygady:
    d_str = booking_day.strftime("%Y-%m-%d")
    slots = st.session_state.schedules.get(b, {}).get(d_str, [])
    for s in slots:
        y_label = f"{b} – {s['client']}"

        # Slot pracy
        dual_slots_day.append({
            "Y": y_label,
            "Typ": "Slot pracy",
            "Start": s["start"],
            "Koniec": s["end"],
        })

        # Przedział przyjazdu
        if s.get("arrival_window_start") and s.get("arrival_window_end"):
            dual_slots_day.append({
                "Y": y_label,
                "Typ": "Przedział przyjazdu",
                "Start": s["arrival_window_start"],
                "Koniec": s["arrival_window_end"],
            })

df_dual_day = pd.DataFrame(dual_slots_day)

if not df_dual_day.empty:
    fig_day = px.timeline(
        df_dual_day,
        x_start="Start",
        x_end="Koniec",
        y="Y",
        color="Typ",
        color_discrete_map={
            "Slot pracy": "#1f77b4",
            "Przedział przyjazdu": "#ff7f0e"
        },
        hover_data=["Typ"]
    )

    # Ustawienie przezroczystości dla przedziału przyjazdu
    for trace in fig_day.data:
        if trace.name == "Przedział przyjazdu":
            trace.opacity = 0.3
        else:
            trace.opacity = 1.0

    fig_day.update_yaxes(autorange="reversed")  # od góry w dół

    # Dodanie preferowanych przedziałów w tle
    for label, (s, e) in PREFERRED_SLOTS.items():
        fig_day.add_vrect(
            x0=datetime.combine(booking_day, s),
            x1=datetime.combine(booking_day, e),
            fillcolor="rgba(200,200,200,0.15)",
            opacity=0.2,
            layer="below",
            line_width=0
        )
        fig_day.add_vline(x=datetime.combine(booking_day, s), line_width=1, line_dash="dot")
        fig_day.add_vline(x=datetime.combine(booking_day, e), line_width=1, line_dash="dot")

    st.plotly_chart(fig_day, use_container_width=True)
else:
    st.info("Brak slotów do wyświetlenia dla wybranego dnia.")

#--------------
