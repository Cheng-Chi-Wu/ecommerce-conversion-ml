"""Optional notebook tooling must execute requested files, never silently skip."""
import pytest

nbformat = pytest.importorskip('nbformat')
nbclient = pytest.importorskip('nbclient')

from src import notebooks


def test_notebook_selection_is_not_shadowed_by_environment_variables(tmp_path, monkeypatch):
    (tmp_path/'notebooks').mkdir()
    (tmp_path/'processed').mkdir()
    for name in ['one.ipynb', 'two.ipynb']:
        nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('print(1)')]),
                       tmp_path/'notebooks'/name)
    monkeypatch.setattr(notebooks, 'ROOT', tmp_path)
    monkeypatch.setattr(notebooks, 'PROCESSED', tmp_path/'processed')
    called = []

    class FakeClient:
        def __init__(self, notebook, **kwargs):
            self.notebook = notebook

        def execute(self):
            called.append(True)
            self.notebook.cells[0].execution_count = 1

    monkeypatch.setattr(nbclient, 'NotebookClient', FakeClient)
    notebooks.execute_notebooks('two.ipynb')
    assert len(called) == 1
    assert nbformat.read(tmp_path/'notebooks/one.ipynb', as_version=4).cells[0].execution_count is None
    assert nbformat.read(tmp_path/'notebooks/two.ipynb', as_version=4).cells[0].execution_count == 1
    with pytest.raises(ValueError, match='No matching notebook'):
        notebooks.execute_notebooks('missing.ipynb')
