import pickle

from widgets.osc_menu import OscMenu


class StubDispatcher:
    def __init__(self):
        self.mapped = []

    def map(self, address, func):
        self.mapped.append((address, func))

    def unmap(self, address, func):
        pass


class StubViz:
    def __init__(self):
        self.osc_dispatcher = StubDispatcher()
        self.errors = []

    def print_error(self, e):
        self.errors.append(e)


def record(address, *args):
    pass


def test_set_params_backfills_missing_keys():
    viz = StubViz()
    old_menu = OscMenu(viz, {"seed": record}, label="##old")
    params = pickle.loads(pickle.dumps(old_menu.get_params()))

    new_menu = OscMenu(viz, {"seed": record, "seed y": record}, label="##new")
    new_menu.set_params(params)

    assert new_menu.use_osc["seed y"] is False
    assert new_menu.osc_addresses["seed y"] == "..."
    assert new_menu.cached_osc_addresses["seed y"] == "..."
    assert new_menu.mappings["seed y"] == "x"
    assert new_menu.use_map["seed y"] is True


def test_set_params_preserves_loaded_values():
    viz = StubViz()
    old_menu = OscMenu(viz, {"seed": record}, label="##old")
    old_menu.use_osc["seed"] = True
    old_menu.osc_addresses["seed"] = "fader1"
    old_menu.cached_osc_addresses["seed"] = "fader1"
    old_menu.mappings["seed"] = "x*20"
    params = pickle.loads(pickle.dumps(old_menu.get_params()))

    new_menu = OscMenu(viz, {"seed": record, "seed y": record}, label="##new")
    new_menu.set_params(params)

    assert new_menu.use_osc["seed"] is True
    assert new_menu.osc_addresses["seed"] == "fader1"
    assert new_menu.mappings["seed"] == "x*20"
    mapped_addresses = [address for address, _ in viz.osc_dispatcher.mapped]
    assert "/fader1" in mapped_addresses


def test_backfilled_key_receives_without_error():
    viz = StubViz()
    old_menu = OscMenu(viz, {"seed": record}, label="##old")
    params = pickle.loads(pickle.dumps(old_menu.get_params()))

    new_menu = OscMenu(viz, {"seed": record, "seed y": record}, label="##new")
    new_menu.set_params(params)

    new_menu.funcs["seed y"]("/pad/y", 0.5)
