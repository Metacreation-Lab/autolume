from widgets.latent_widget import LatentWidget


class StubDispatcher:
    def map(self, address, func):
        pass

    def unmap(self, address, func):
        pass


class StubViz:
    def __init__(self):
        self.osc_dispatcher = StubDispatcher()
        self.errors = []

    def print_error(self, e):
        self.errors.append(e)


def make_widget():
    return LatentWidget(StubViz())


def test_coordinates_initialize_as_floats():
    w = make_widget()
    assert isinstance(w.latent.x, float)
    assert isinstance(w.latent.y, float)


def test_seed_handler_stores_fractional_values():
    w = make_widget()
    w.osc_handler("x", float)("/seed", 3.5)
    assert w.latent.x == 3.5
    assert w.viz.errors == []


def test_seed_handler_stays_float_after_int_assignment():
    w = make_widget()
    w.latent.x = 7
    w.osc_handler("x", float)("/seed", 2.25)
    assert w.latent.x == 2.25
    assert isinstance(w.latent.x, float)


def test_seed_y_handler_writes_y():
    w = make_widget()
    w.osc_handler("y", float)("/pad/y", 1.75)
    assert w.latent.y == 1.75


def test_seed_menu_keys():
    w = make_widget()
    assert list(w.seed_osc_menu.funcs.keys()) == [
        "project", "seed", "seed y", "anim", "speed", "model"]


def test_vec_menu_has_no_seed_keys():
    w = make_widget()
    assert "seed" not in w.vec_osc_menu.funcs
    assert "seed y" not in w.vec_osc_menu.funcs


def test_seed_y_end_to_end_through_menu():
    w = make_widget()
    w.seed_osc_menu.use_osc["seed y"] = True
    w.seed_osc_menu.funcs["seed y"]("/pad/y", 0.6)
    assert w.latent.y == 0.6
