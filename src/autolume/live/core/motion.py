"""Wall-clock motion integration, run on the control tick."""

import dataclasses

from autolume.live.core.params import ControlState


def integrate(state: ControlState, dt: float) -> ControlState:
    if not state.anim_playing or dt <= 0.0:
        return state
    return dataclasses.replace(
        state,
        latent_x=state.latent_x + state.anim_speed_x * dt,
        latent_y=state.latent_y + state.anim_speed_y * dt,
    )
