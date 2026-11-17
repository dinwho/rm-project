"""A legal, hand-built boundary barrier versus waiting on identical random draws.

This deliberately fixed example proves a physical effect, not typical policy
performance. No random draws, fuel cells or fire states are modified externally.
The controlled full-ring and diagonal-gap stress tests are in tests/environment/.
"""
from wildfire_crmab import EnvironmentConfig, Scenario, WildfireEnv


def episode(construct):
    env = WildfireEnv(EnvironmentConfig(fixed_ignition=(0, 0), fixed_wind=(0.0, 0.8)))
    env.reset(options={"scenario": Scenario(seed=0, budget=None)})
    # The outer boundary and these adjacent cleared cells separate corner fire
    # from unburned vegetation, including the diagonal (1,1) escape route.
    for row, col in ((1, 0), (1, 1), (0, 1)):
        action = env.allocations.encode((row*env.config.grid_size+col,)) if construct else 0
        env.step(action)  # ONE cell, then ONE fire update; never prebuild a barrier.
    while not env.finished:
        env.step(0)
    return env


if __name__ == "__main__":
    for build in (False, True):
        env = episode(build)
        print(f"{'Hand-built barrier' if build else 'Wait only'}: "
              f"burned={env.ever_burned.sum()}, cleared={env.firebreak.sum()}, "
              f"affected={env.info()['affected_fraction']:.1%}, ticks={env.step_number}")
    assert episode(True).ever_burned.sum() == 1
