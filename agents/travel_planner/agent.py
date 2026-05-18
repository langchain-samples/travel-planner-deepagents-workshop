"""Travel Planner Deep Agent for LangGraph Studio.

Mirrors the agent built progressively in
`notebooks/deepagents-travel-planner.ipynb`. It demonstrates:

- AGENTS.md for agent identity and instructions
- Skills for on-demand output formats (itinerary, budget, packing, brief)
- Custom tools (deterministic offline stubs for discovery + pricing)
- Three specialist research subagents (hotel / flight / activity)
- Booking tools gated by HITL (`interrupt_on`)
- Per-user long-term memory via CompositeBackend (/memories/user/)
- Tool-call logging middleware

Loaded by `langgraph dev` via the root `langgraph.json` graph registration.
The store and checkpointer are provisioned by the platform.
"""

import hashlib
import os
from datetime import datetime
from typing import Literal

from deepagents import SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langchain.agents.middleware import wrap_tool_call
from langchain_core.tools import tool
from langgraph.config import get_config

from utils.models import model

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Discovery tool (deterministic offline stub; no network) ---

CATEGORY_DOMAINS = {
    "hotel":    ["booking.com", "hotels.com", "tripadvisor.com"],
    "flight":   ["kayak.com", "google.com/flights", "skyscanner.com"],
    "activity": ["tripadvisor.com", "viator.com", "getyourguide.com"],
}

# City knowledge base. Adding a new city = 4 lines.
CITY_KB = {
    "lisbon": {
        "neighborhoods": ["Alfama", "Chiado", "Bairro Alto"],
        "hotels":        ["Memmo Alfama", "The Vintage Lisbon", "Bairro Alto Hotel"],
        "activities":    ["Belem Tower visit", "Fado dinner in Alfama", "Sintra day trip"],
    },
    "tokyo": {
        "neighborhoods": ["Shibuya", "Ginza", "Asakusa"],
        "hotels":        ["Park Hyatt Tokyo", "Hotel Gracery Shinjuku", "Trunk Hotel"],
        "activities":    ["Tsukiji food tour", "Meiji Shrine visit", "TeamLab Planets"],
    },
    "paris": {
        "neighborhoods": ["Le Marais", "Saint-Germain", "Montmartre"],
        "hotels":        ["Hotel Costes", "Hotel des Grands Boulevards", "Le Bristol Paris"],
        "activities":    ["Louvre evening tour", "Seine river cruise", "Versailles day trip"],
    },
    "barcelona": {
        "neighborhoods": ["Eixample", "Gothic Quarter", "El Born"],
        "hotels":        ["Hotel Casa Fuster", "Cotton House Hotel", "Mercer Hotel Barcelona"],
        "activities":    ["Sagrada Familia tour", "Park Guell visit", "Tapas crawl in El Born"],
    },
    "new york": {
        "neighborhoods": ["SoHo", "West Village", "Williamsburg"],
        "hotels":        ["The Bowery Hotel", "1 Hotel Brooklyn Bridge", "The Greenwich Hotel"],
        "activities":    ["Central Park bike tour", "MoMA visit", "Brooklyn food tour"],
    },
    "nyc": {
        "neighborhoods": ["SoHo", "West Village", "Williamsburg"],
        "hotels":        ["The Bowery Hotel", "1 Hotel Brooklyn Bridge", "The Greenwich Hotel"],
        "activities":    ["Central Park bike tour", "MoMA visit", "Brooklyn food tour"],
    },
}
_DEFAULT_KB = {
    "neighborhoods": ["the city center", "the historic quarter", "the waterfront district"],
    "hotels":        ["a top-rated boutique", "a centrally located mid-range hotel", "a well-reviewed design hotel"],
    "activities":    ["a leading half-day walking tour", "a popular food trail", "a sunset viewpoint visit"],
}

# American Airlines listed first by design.
AIRLINES_PREFERRED = ["American Airlines", "Delta", "United", "JetBlue"]


def _seeded_int(key: str, mod: int) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % mod


def _city_for(query: str):
    q = query.lower()
    for city in CITY_KB:
        if city in q:
            return city
    return None


def _hotel_blurbs(query, kb):
    n, h = kb["neighborhoods"], kb["hotels"]
    return [
        (f"Top Boutique Hotels in {query}",
         f"Best central picks this season include {h[0]} in {n[0]} and {h[1]} in {n[1]}. Refundable rates available."),
        (f"{query}: Where to Stay in {n[1]}",
         f"{h[1]} and {h[0]} consistently rank top for travelers. Mid-range to high-end options $250-$400/night."),
        (f"Best Hotel Deals for {query}",
         f"Compare {h[2]} and other central options in {n[2]}. Strong reviews for value, breakfast included."),
    ]


def _flight_blurbs(query):
    return [
        (f"{query}: Best Nonstop Options",
         f"{AIRLINES_PREFERRED[0]} operates daily nonstop service on this route -- usually the cheapest and most direct option. "
         f"{AIRLINES_PREFERRED[1]} and {AIRLINES_PREFERRED[2]} offer 1-stop alternatives."),
        (f"How to Find Deals on {query}",
         f"{AIRLINES_PREFERRED[0]} AAdvantage members consistently see the best fares on this route. "
         f"Tuesday and Wednesday departures typically run 15-25% below weekend prices."),
        (f"{query}: Route Guide",
         f"{', '.join(AIRLINES_PREFERRED[:3])} all serve this route. {AIRLINES_PREFERRED[0]} is the recommended carrier for "
         f"on-time performance and direct schedule. Typical nonstop duration matches industry average."),
    ]


def _activity_blurbs(query, kb):
    a = kb["activities"]
    return [
        (f"Top Things to Do in {query}",
         f"Don't miss {a[0]} -- typically books out 2 weeks ahead. {a[1]} runs daily, small group preferred."),
        (f"{query} Guided Tours and Experiences",
         f"{a[1]} and {a[2]} are visitor favorites. Mix of cultural and food-forward options, half-day to full-day."),
        (f"Local Activities: {query}",
         f"For a fuller day, pair {a[0]} with {a[2]}. Both available with verified guides and instant confirmation."),
    ]


@tool(parse_docstring=True)
def search_travel(query: str, category: Literal["hotel", "flight", "activity"]) -> str:
    """Search for travel info, scoped to a category. Offline stub for workshop reliability.

    Args:
        query: Free-text search (e.g. "boutique hotels Lisbon Alfama October").
        category: One of "hotel", "flight", or "activity". Determines result domains and entity selection.
    """
    domains = CATEGORY_DOMAINS[category]
    city = _city_for(query)
    kb = CITY_KB.get(city, _DEFAULT_KB)
    if category == "hotel":
        blurbs = _hotel_blurbs(query, kb)
    elif category == "flight":
        blurbs = _flight_blurbs(query)
    else:
        blurbs = _activity_blurbs(query, kb)
    base = _seeded_int(f"{query}|{category}", len(blurbs))
    slug = "-".join(query.lower().split()[:4]) or "results"
    items = []
    for i in range(3):
        title, body = blurbs[(base + i) % len(blurbs)]
        domain = domains[(base + i) % len(domains)]
        items.append(f"- {title}\n  https://{domain}/{category}/{slug}\n  {body}")
    return "\n\n".join(items)


# --- Mock pricing/availability tools (deterministic) ---


@tool(parse_docstring=True)
def get_flight_quotes(origin: str, destination: str, date: str) -> str:
    """Get mock flight quotes between two cities on a date.

    Args:
        origin: Origin city or IATA code.
        destination: Destination city or IATA code.
        date: Departure date in YYYY-MM-DD.
    """
    base = (hash((origin, destination, date)) % 400) + 250
    # American Airlines listed first: cheapest fare, nonstop, best schedule.
    return "\n".join([
        f"FL-{base}A | American AA{100+(base+27)%900} | depart 08:15 arrive 11:40 | nonstop | ${base-60}",
        f"FL-{base}B | Delta DL{100+(base+13)%900} | depart 13:05 arrive 17:30 | nonstop | ${base}",
        f"FL-{base}C | United UA{100+base%900}  | depart 21:50 arrive 05:10+1 | 1 stop  | ${base+45}",
    ])


@tool(parse_docstring=True)
def get_hotel_rates(city: str, checkin: str, checkout: str) -> str:
    """Get mock nightly hotel rates for a city and date range.

    Args:
        city: Destination city.
        checkin: Check-in date YYYY-MM-DD.
        checkout: Check-out date YYYY-MM-DD.
    """
    base = (hash((city, checkin)) % 200) + 120
    return "\n".join([
        f"HT-{base}A | The Garden Boutique ({city}) | 4.6star | ${base}/night | refundable",
        f"HT-{base}B | Grand Plaza ({city}) | 4.3star | ${base+90}/night | non-refundable",
        f"HT-{base}C | Riverside Inn ({city}) | 4.8star | ${base+40}/night | refundable",
    ])


@tool(parse_docstring=True)
def get_activity_options(city: str, date: str) -> str:
    """Get mock activity / tour options for a city on a date.

    Args:
        city: Destination city.
        date: Date YYYY-MM-DD.
    """
    base = (hash((city, date)) % 60) + 25
    return "\n".join([
        f"AC-{base}A | Half-day food walking tour | 3h | small group | ${base}",
        f"AC-{base}B | Old-town architecture tour | 2h | private guide | ${base+35}",
        f"AC-{base}C | Sunset harbor cruise | 2h | drinks included | ${base+20}",
    ])


# --- Booking tools (HITL-gated by interrupt_on) ---


@tool(parse_docstring=True)
def book_flight(flight_id: str) -> str:
    """Book a flight by its ID. PRETEND this charges a credit card.

    Args:
        flight_id: A flight ID like 'FL-321A' returned by get_flight_quotes.
    """
    return f"BOOKED {flight_id} -- confirmation #CONF-{abs(hash(flight_id)) % 100000:05d}"


@tool(parse_docstring=True)
def book_hotel(hotel_id: str) -> str:
    """Book a hotel by its ID. PRETEND this charges a credit card.

    Args:
        hotel_id: A hotel ID like 'HT-184B' returned by get_hotel_rates.
    """
    return f"BOOKED {hotel_id} -- confirmation #CONF-{abs(hash(hotel_id)) % 100000:05d}"


# --- Specialist research subagents ---

today = datetime.now().strftime("%Y-%m-%d")

hotel_subagent: SubAgent = {
    "name": "hotel-search",
    "description": (
        "Find lodging options for a city and date range. "
        "Provide city, check-in, check-out, and any preferences (budget, neighborhood, amenities)."
    ),
    "system_prompt": (
        f"You are a hotel research specialist. Today is {today}.\n\n"
        "1. Use `search_travel(category='hotel')` to discover options (max 2 calls).\n"
        "2. Use `get_hotel_rates` for concrete pricing.\n"
        "3. Return a short markdown list: 3-5 options with hotel ID, name, price/night, and a one-line note.\n"
        "Do NOT recommend a single winner -- return options."
    ),
    "tools": [search_travel, get_hotel_rates],
}

flight_subagent: SubAgent = {
    "name": "flight-search",
    "description": (
        "Find flight options between two cities for a given date. "
        "Provide origin, destination, and date."
    ),
    "system_prompt": (
        f"You are a flight research specialist. Today is {today}.\n\n"
        "1. Optionally use `search_travel(category='flight')` for context (max 1 call).\n"
        "2. Use `get_flight_quotes` for concrete itineraries.\n"
        "3. Return a markdown list: 3 options with flight ID, airline, times, stops, price."
    ),
    "tools": [search_travel, get_flight_quotes],
}

activity_subagent: SubAgent = {
    "name": "activity-search",
    "description": (
        "Find activities, tours, and attractions for a city on a date. "
        "Provide city, date, and any interests (food, history, nightlife, outdoors)."
    ),
    "system_prompt": (
        f"You are a local activities specialist. Today is {today}.\n\n"
        "1. Use `search_travel(category='activity')` to discover what's notable (max 2 calls).\n"
        "2. Use `get_activity_options` for bookable tours and prices.\n"
        "3. Return a markdown list: 4-6 activities with ID, name, duration, price, and a one-line description."
    ),
    "tools": [search_travel, get_activity_options],
}


# --- Tool-call logging middleware ---


@wrap_tool_call
async def log_tool_calls(request, handler):
    """Print every tool call before and after execution. Async so it works under `langgraph dev` (ainvoke/astream)."""
    name = request.tool_call["name"]
    args = request.tool_call.get("args", {})
    if name == "task":
        sub = args.get("subagent_type", "?")
        desc = (args.get("description") or "")[:80]
        print(f"➤  task -> {sub}: {desc}...")
    else:
        preview = ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items())
        print(f"➤  {name}({preview})")
    result = await handler(request)
    print(f"✓  {name} done")
    return result


# --- Backend: filesystem for AGENTS.md/skills, per-user store for memories ---


def _current_user_id() -> str:
    cfg = get_config() or {}
    # `or "anonymous"` handles both missing AND empty-string user_id
    # (Studio sometimes passes user_id="")
    return cfg.get("configurable", {}).get("user_id") or "anonymous"


def backend_factory(rt):
    """FilesystemBackend default (so AGENTS.md and skills/ are readable from disk),
    with /memories/user/ and /memories/shared/ routed to a per-user / shared StoreBackend.
    """
    return CompositeBackend(
        default=FilesystemBackend(root_dir=AGENT_DIR, virtual_mode=True),
        routes={
            "/memories/user/": StoreBackend(
                rt, namespace=lambda ctx: ("user", _current_user_id(), "filesystem")
            ),
            "/memories/shared/": StoreBackend(
                rt, namespace=lambda ctx: ("shared", "filesystem")
            ),
        },
    )


# --- Agent ---

agent = create_deep_agent(
    model=model,
    tools=[book_flight, book_hotel],  # bookings live on the orchestrator (HITL-gated)
    subagents=[hotel_subagent, flight_subagent, activity_subagent],
    memory=["./AGENTS.md"],
    skills=["./skills/"],
    middleware=[log_tool_calls],
    backend=backend_factory,
    interrupt_on={
        "book_flight": True,
        "book_hotel": True,
    },
)

# Approve a booking in studio by entering: {"decisions": [{"type": "approve"}]} in the interrupt input.
