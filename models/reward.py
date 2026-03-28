from pydantic import BaseModel


class Reward(BaseModel):
    total: float                   # final scalar reward passed to agent
    safety_component: float        # contribution from safety
    time_component: float          # contribution from time efficiency
    fuel_component: float          # contribution from fuel efficiency
    penalty: float                 # any penalties applied this step
    reason: str                    # human-readable explanation
