"""Regression checks for the error-calculation merge resolution."""
import builtins
import io

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from mlputils import benchmark as b


class Model(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, stress=True):
        super().__init__()
        self.calls = 0
        if not stress:
            self.implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert len(atoms) > 1, 'The supplied isolated-atom energy must be reused'
        self.calls += 1
        self.results = dict(energy=-22.0 * len(atoms),
                            forces=np.full((len(atoms), 3), .2))
        if 'stress' in self.implemented_properties:
            self.results['stress'] = np.full(6, .2)


def reference_frames(stress=True):
    frames = []
    for n, energy, value in [(2, -13., .1), (3, -15., .4)]:
        atoms = Atoms('Cu' * n, positions=np.arange(n * 3).reshape(n, 3), cell=[20] * 3)
        results = dict(energy=energy * n, forces=np.full((n, 3), value))
        if stress:
            results['stress'] = np.full(6, value)
        atoms.calc = SinglePointCalculator(atoms, **results)
        frames.append(atoms)
    return frames


@pytest.mark.parametrize('metric', ['mae', 'rmse'])
@pytest.mark.parametrize('norm', [False, True])
@pytest.mark.parametrize('stress', ['both', 'no_reference', 'no_model'])
def test_metrics_and_repeat_evaluation(tmp_path, metric, norm, stress):
    frames = reference_frames(stress != 'no_reference')
    original_calculators = [frame.calc for frame in frames]
    model = Model(stress != 'no_model')
    expected_e = 2. if metric == 'mae' else np.sqrt(5.)
    expected_f = .15 if metric == 'mae' else np.sqrt(.025)
    expected_s = expected_f
    if norm:
        expected_f *= np.sqrt(3)
        expected_s *= 3
    for _ in range(2):
        result = b.compute_test_errors(model, {'offline_test': frames}, -10., -20.,
                                       err_method=metric, use_norm=norm,
                                       out_folder=str(tmp_path))[0]['offline_test']
        assert result['energy_per_atom'][metric] == pytest.approx(expected_e)
        assert result['forces'][metric] == pytest.approx(expected_f)
        if stress == 'both':
            assert result['stress'][metric] == pytest.approx(expected_s)
        else:
            assert result['stress'] is None
        assert [frame.calc for frame in frames] == original_calculators
        assert [frame.get_potential_energy() for frame in frames] == [-26., -45.]
        np.testing.assert_allclose(frames[0].get_forces(), .1)
        np.testing.assert_allclose(frames[1].get_forces(), .4)
        rows = (tmp_path / '_offline_test_errors.dat').read_text().splitlines()
        assert len(rows) == 3  # Header and one row per reference configuration.
        assert [row.split()[0] for row in rows[1:]] == ['0', '1']
        assert len((tmp_path / '_offline_test_e-parity.dat').read_text().splitlines()) == 2
        force_rows = np.loadtxt(tmp_path / '_offline_test_f-parity.dat')
        assert force_rows.shape == ((5, 2) if norm else (15, 2))
    assert model.calls == 4  # One prediction per frame, per evaluation.


def test_stress_count_resets_between_datasets(tmp_path):
    result = b.compute_test_errors(Model(), {'with_stress': reference_frames(),
                                            'without_stress': reference_frames(False)},
                                   -10., -20., out_folder=str(tmp_path))
    assert result[0]['with_stress']['stress']['mae'] == pytest.approx(.45)
    assert result[1]['without_stress']['stress'] is None


def test_file_paths_and_same_basename_outputs(tmp_path):
    files = []
    for folder in ['a', 'b']:
        path = tmp_path / folder / 'test.xyz'
        path.parent.mkdir()
        write(path, reference_frames())
        files.append(str(path))
    output = tmp_path / 'nested' / 'results'
    results = b.compute_test_errors(Model(), files, -10., -20., out_folder=str(output))
    assert [next(iter(result)) for result in results] == files
    assert len(list(output.glob('*errors.dat'))) == 2
    for result in results:
        assert next(iter(result.values()))['energy_per_atom']['mae'] == pytest.approx(2.)
    for file in output.glob('*errors.dat'):
        assert len(file.read_text().splitlines()) == 3
    # The single-filename input form is also used by benchmark.main().
    result = b.compute_test_errors(Model(), files[0], -10., -20., out_folder=str(output))
    assert result[0][files[0]]['energy_per_atom']['mae'] == pytest.approx(2.)


@pytest.mark.parametrize('fail', [False, True])
def test_streams_close_on_success_and_prediction_failure(tmp_path, monkeypatch, fail):
    handles = []

    def tracking_open(*args, **kwargs):
        handle = builtins.open(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(b, 'open', tracking_open, raising=False)
    model = Model()
    if fail:
        def broken_prediction(*args, **kwargs):
            raise RuntimeError('prediction failed')
        monkeypatch.setattr(model, 'calculate', broken_prediction)
        with pytest.raises(RuntimeError, match='prediction failed'):
            b.compute_test_errors(model, {'sample': reference_frames()}, -10., -20.,
                                  out_folder=str(tmp_path))
    else:
        b.compute_test_errors(model, {'sample': reference_frames()}, -10., -20.,
                              out_folder=str(tmp_path))
    assert len(handles) == 4
    assert all(handle.closed for handle in handles)


def test_benchmark_main_saves_error_metrics(tmp_path, monkeypatch):
    dataset = tmp_path / 'test.xyz'
    write(dataset, reference_frames(False))
    output = tmp_path / 'results'
    setup = dict(symbol='Cu', E_iso=-10., fcc_lattice_constant=3.6,
                 cohesive_energy=-3., Bulk_modulus=100., calculator='test_model',
                 test_set_file=str(dataset), output_folder=str(output),
                 **{'111_surface_energy': 1., '110_surface_energy': 1.,
                    '100_surface_energy': 1.})
    config = tmp_path / 'benchmark.yaml'
    config.write_text(yaml.safe_dump(setup))
    monkeypatch.setattr(b, 'get_calc', lambda config: Model())
    monkeypatch.setattr(b, 'get_isolated_atom_energy', lambda *args: -20.)
    monkeypatch.setattr(b, 'eos_fcc_fit', lambda *args, **kwargs:
                        {'fcc_bulk': dict(a0=3.6, e0=-3., B=100.)})
    monkeypatch.setattr(b, 'low_index_surfen', lambda *args, **kwargs:
                        {name: {'gamma': 1.} for name in ['fcc111', 'fcc110', 'fcc100']})
    b.main(str(config))
    saved = yaml.safe_load((output / 'test_model_benchmark.yaml').read_text())
    assert saved[str(dataset)]['energy_per_atom']['mae'] == pytest.approx(2.)
    assert saved[str(dataset)]['stress'] is None


def test_offline_error_logging_accepts_in_memory_dataset(tmp_path):
    from mlputils.offline_training import log_errors_gibo

    errors = b.compute_test_errors(Model(), {'offline_test': reference_frames(False)},
                                   -10., -20., use_norm=False, out_folder=str(tmp_path))
    energy_log, force_log = io.StringIO(), io.StringIO()
    log_errors_gibo(errors, energy_log, force_log, 7)
    assert energy_log.getvalue().split() == ['7', '2.0']
    step, force_mae = force_log.getvalue().split()
    assert step == '7'
    assert float(force_mae) == pytest.approx(.15)
