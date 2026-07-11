from daedalus.governor import GovernorConfig, LevelPolicy, ThermalGovernor
from daedalus.sensors import ThermalLevel, ThermalMonitor


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_monitor(initial=ThermalLevel.NOMINAL):
    state = {"level": initial}
    monitor = ThermalMonitor(reader=lambda: state["level"], poll_interval=1000)
    return monitor, state


def make_governor(initial=ThermalLevel.NOMINAL, **cfg):
    monitor, state = make_monitor(initial)
    clock = FakeClock()
    gov = ThermalGovernor(monitor, GovernorConfig(**cfg), clock=clock)

    def set_level(level):
        state["level"] = level
        monitor.refresh()

    return gov, set_level, clock


def test_nominal_runs_full_speed():
    gov, _, _ = make_governor()
    d = gov.pace(chunk_seconds=5.0)
    assert d.next_chunk_tokens == 2048
    assert d.sleep_seconds == 0.0
    assert d.effective_level == ThermalLevel.NOMINAL


def test_escalation_is_instant():
    gov, set_level, _ = make_governor()
    set_level(ThermalLevel.HEAVY)
    d = gov.pace(chunk_seconds=2.0)
    assert d.effective_level == ThermalLevel.HEAVY
    assert d.next_chunk_tokens == 512
    # duty 0.25 -> sleep = 2.0 * 0.75/0.25 = 6.0
    assert abs(d.sleep_seconds - 6.0) < 1e-9


def test_sleep_is_capped():
    gov, set_level, _ = make_governor()
    set_level(ThermalLevel.HEAVY)
    d = gov.pace(chunk_seconds=60.0)
    assert d.sleep_seconds == 10.0  # max_sleep_seconds default


def test_deescalation_requires_dwell_time():
    gov, set_level, clock = make_governor(initial=ThermalLevel.HEAVY)
    set_level(ThermalLevel.NOMINAL)

    d = gov.pace(chunk_seconds=1.0)
    assert d.effective_level == ThermalLevel.HEAVY  # not yet

    clock.advance(10.0)
    assert gov.pace(1.0).effective_level == ThermalLevel.HEAVY  # still dwelling

    clock.advance(11.0)  # total 21s below > step_down_seconds=20
    assert gov.pace(1.0).effective_level == ThermalLevel.MODERATE  # one step only

    clock.advance(21.0)
    assert gov.pace(1.0).effective_level == ThermalLevel.NOMINAL


def test_reescalation_resets_dwell():
    gov, set_level, clock = make_governor(initial=ThermalLevel.HEAVY)
    set_level(ThermalLevel.MODERATE)
    gov.pace(1.0)
    clock.advance(15.0)
    set_level(ThermalLevel.HEAVY)  # heats up again before dwell elapsed
    gov.pace(1.0)
    set_level(ThermalLevel.MODERATE)
    clock.advance(15.0)  # only 15s below since re-escalation
    assert gov.pace(1.0).effective_level == ThermalLevel.HEAVY


def test_quiet_mode_caps_duty_even_when_cool():
    gov, _, _ = make_governor(max_duty=0.5)
    d = gov.pace(chunk_seconds=4.0)
    assert d.effective_level == ThermalLevel.NOMINAL
    assert abs(d.sleep_seconds - 4.0) < 1e-9  # duty 0.5 -> idle == burn


def test_custom_policies():
    policies = dict(
        {
            level: LevelPolicy(chunk_tokens=100, duty=1.0)
            for level in ThermalLevel
        }
    )
    gov, _, _ = make_governor(policies=policies)
    assert gov.pace(1.0).next_chunk_tokens == 100
    assert gov.initial_chunk_tokens() == 100
