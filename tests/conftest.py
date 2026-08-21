"""Shared test setup.

CI runs headless, where PyOpenGL cannot bind a GL library; importing any
widget module (they pull in utils.gui_utils.gl_utils) would then fail at
collection. The widget tests only exercise GL-free logic, so when the real
binding is unavailable a permissive stub is installed before the test
modules import.
"""
import sys
import types

try:
    import OpenGL.GL  # noqa: F401
except Exception:
    class _StubModule(types.ModuleType):
        """OpenGL stand-in: GL_* constants are ints, functions are no-ops."""
        def __getattr__(self, name):
            if name.startswith('__'):
                raise AttributeError(name)
            if name.isupper():
                return 0
            return lambda *args, **kwargs: None

    def _install(name):
        module = _StubModule(name)
        sys.modules[name] = module
        parent, _, child = name.rpartition('.')
        if parent:
            setattr(sys.modules[parent], child, module)

    for _name in ('OpenGL', 'OpenGL.GL', 'OpenGL.GL.ARB',
                  'OpenGL.GL.ARB.texture_float', 'OpenGL.EGL'):
        _install(_name)
