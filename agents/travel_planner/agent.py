"""Travel Planner Deep Agent for LangGraph Studio / Platform.

A travel planning agent assembled from the patterns built up in
`notebooks/cisco/deep-agents.ipynb`. It demonstrates:

- AGENTS.md for agent identity and instructions
- Skills for on-demand output formats (itinerary, budget, packing, brief)
- Custom tools (Tavily-backed discovery + mock pricing)
- Three specialist research subagents (hotel / flight / activity)
- Booking tools gated by HITL (`interrupt_on`)
- Per-user long-term memory via CompositeBackend (/memories/user/)
- Tool-call logging middleware

When running via `langgraph dev` or `deepagents deploy`, the store and
checkpointer are provisioned by the platform.
"""

import os
from datetime import datetime
from typing import Literal

from deepagents import SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langchain.agents.middleware import wrap_tool_call
from langchain_core.tools import tool
from langgraph.config import get_config
from tavily import TavilyClient

from utils.models import model

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Discovery tool (real web search, scoped by category) ---

tavily_client = TavilyClient()

CATEGORY_SITES = {
    "hotel": "site:booking.com OR site:hotels.com OR site:tripadvisor.com",
    "flight": "site:kayak.com OR site:google.com/flights OR site:skyscanner.com",
    "activity": "site:tripadvisor.com OR site:viator.com OR site:getyourguide.com",
}


@tool(parse_docstring=True)
def search_travel(query: str, category: Literal["hotel", "flight", "activity"]) -> str:
    """Search the web for travel info, scoped to a category.

    Args:
        query: Free-text search (e.g. "boutique hotels Lisbon Alfama October").
        category: One of "hotel", "flight", or "activity". Determines which sites are searched.
    """
    scoped = f"{query} {CATEGORY_SITES[category]}"
    results = tavily_client.search(scoped, max_results=3, topic="general")
    out = []
    for r in results.get("results", []):
        out.append(f"- {r['title']}\n  {r['url']}\n  {r['content'][:300]}")
    return "\n\n".join(out) if out else "No results."


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
    return "\n".join([
        f"FL-{base}A | United UA{100+base%900} | depart 08:15 arrive 11:40 | nonstop | ${base}",
        f"FL-{base}B | Delta DL{100+(base+13)%900} | depart 13:05 arrive 17:30 | nonstop | ${base+45}",
        f"FL-{base}C | American AA{100+(base+27)%900} | depart 21:50 arrive 05:10+1 | 1 stop | ${base-60}",
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
def log_tool_calls(request, handler):
    """Print every tool call before and after execution."""
    name = request.tool_call["name"]
    args = request.tool_call.get("args", {})
    if name == "task":
        sub = args.get("subagent_type", "?")
        desc = (args.get("description") or "")[:80]
        print(f"➤  task -> {sub}: {desc}...")
    else:
        preview = ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items())
        print(f"➤  {name}({preview})")
    result = handler(request)
    print(f"✓  {name} done")
    return result


# --- Backend: filesystem for AGENTS.md/skills, per-user store for memories ---


def _current_user_id() -> str:
    cfg = get_config() or {}
    return cfg.get("configurable", {}).get("user_id", "anonymous")


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
