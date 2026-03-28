from pydantic import BaseModel
from typing import Optional, Literal


class Action(BaseModel):
    action_type: Literal[
        "select_route",    # choose a route to follow
        "reroute",         # switch to a different route mid-journey
        "report_incident", # report a new incident
        "continue",        # keep going on current route
        "stop"             # abandon journey
    ]
    route_id: Optional[str] = None          # for select_route / reroute
    incident_type: Optional[str] = None     # for report_incident
    incident_lat: Optional[float] = None
    incident_lng: Optional[float] = None
