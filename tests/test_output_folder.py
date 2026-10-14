from pathlib import Path
import numpy as np
import pytest,yaml
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from flare.bffs.sgp.calculator import SGP_Calculator
from mlputils import offline_training as o,benchmark as b,reference_dft_maker as d,makesets as m
from mlputils._outputs import in_output_folder
ROOT=Path(__file__).resolve().parents[1]

def folder_config(tmp_path,mode):
 if mode=='default':return {},tmp_path
 folder='nested/results' if mode=='relative' else str(tmp_path/'absolute'/'results')
 return {'output_folder':folder},(tmp_path/folder)

def frames():
 result=[]
 for shift in [0.,.1,.2]:
  a=bulk('Cu','fcc',a=3.6,cubic=True);a.positions[0,0]+=shift;a.calc=EMT()
  a.calc=SinglePointCalculator(a,energy=a.get_potential_energy(),forces=a.get_forces(),stress=a.get_stress())
  result.append(a)
 return result

@pytest.mark.parametrize('mode',['default','relative','absolute'])
def test_offline_main_outputs(tmp_path,monkeypatch,mode):
 monkeypatch.chdir(tmp_path)
 setting,folder=folder_config(tmp_path,mode)
 data=frames();write('input-train.xyz',data);write('input-test.xyz',[data[-1]])
 cfg=yaml.safe_load((ROOT/'workflows_examples/offline_setup.yaml').read_text())
 cfg.pop('output_folder', None)
 cfg.update(setting,dataset_style='intact',train_file='input-train.xyz',test_file='input-test.xyz',files_prefix='outputcheck',nr_initial_envs=2,min_optimize=100,max_optimize=200,call_threshold=0.,add_threshold=1e-15)
 cfg['flare_calc']['descriptors'][0].update(nmax=2,lmax=1)
 cfg['flare_calc'].update(single_atom_energies=[0.])
 Path('setup.yaml').write_text(yaml.safe_dump(cfg))
 o.main('setup.yaml')
 expected=['training_outputcheck.log','hyps_outputcheck.dat','lik_outputcheck.dat','e_maes_outputcheck.dat','f_maes_outputcheck.dat','outputcheck_model.json','added_structures.xyz','lmp.flare','L_inv_lmp.flare','sparse_desc_lmp.flare','_offline_test_errors.dat']
 for name in expected:assert (folder/name).is_file(),name
 restored,kernels=SGP_Calculator.from_file(str(folder/'outputcheck_model.json'))
 a=data[-1].copy();a.calc=restored;assert np.isfinite(a.get_potential_energy())
 assert Path.cwd()==tmp_path
 if mode!='default':assert {p.name for p in tmp_path.iterdir() if p.is_file()}=={'input-train.xyz','input-test.xyz','setup.yaml'}
 # Dataset construction also respects the same folder for generated train/test splits.
 m.make_sets(dict(setting,dataset_style='random',bulk_files=['input-train.xyz'],test_set_split=.34))
 assert (folder/'train.xyz').exists() and (folder/'test.xyz').exists()

@pytest.mark.parametrize('mode',['default','relative','absolute'])
def test_benchmark_main_outputs(tmp_path,monkeypatch,mode):
 monkeypatch.chdir(tmp_path);setting,folder=folder_config(tmp_path,mode)
 write('reference.xyz',[frames()[-1]])
 cfg=dict(setting,symbol='Cu',E_iso=0.,fcc_lattice_constant=3.6,cohesive_energy=-3.,Bulk_modulus=130.,**{'111_surface_energy':1.,'110_surface_energy':1.,'100_surface_energy':1.},calculator='emt',test_set_file='reference.xyz',special_configs=['reference.xyz'],special_configs_folder='./')
 Path('benchmark.yaml').write_text(yaml.safe_dump(cfg))
 monkeypatch.setattr(b,'get_calc',lambda config:EMT())
 b.main('benchmark.yaml')
 for name in ['emt_benchmark.yaml','eos.dat','bfgs.log','_reference.xyz_errors.dat']:assert (folder/name).exists()
 if mode!='default':assert {p.name for p in tmp_path.iterdir() if p.is_file()}=={'benchmark.yaml','reference.xyz'}

@pytest.mark.parametrize('mode',['default','relative','absolute'])
def test_dft_inputs(tmp_path,monkeypatch,mode):
 monkeypatch.chdir(tmp_path);setting,folder=folder_config(tmp_path,mode)
 params=dict(setting,ecutwfc=45,dual=8,degauss_eV=.1,econv_eV_peratom=1e-6,pseudos={'Cu':'Cu.UPF'},kpts=14,latticeconstant={'Cu':3.6})
 Path('parameters.yml').write_text(yaml.safe_dump(params))
 relaxed=dict(params);relaxed.pop('output_folder',None)
 Path('parameters_relax.yml').write_text(yaml.safe_dump(relaxed))
 d.write_all_inputs(['reference_dft_maker','input','Cu','parameters.yml'])
 for name in ['Cu_iso.pwi','Cu_fcc_relax.pwi','Cu_surf_111_1x1x8_relax.pwi','Cu_Ih_55_relax.pwi']:assert (folder/name).exists()
 d.input_phonons('Cu',3.6,'pseudos',params,supercell_size=2)
 assert (folder/'Cu_phonopy.yaml').exists() and (folder/'Cu_phonopy_structure_0.pwi').exists()
 if mode!='default':assert {p.name for p in tmp_path.iterdir() if p.is_file()}=={'parameters.yml','parameters_relax.yml'}

def test_third_party_writer_restores_cwd_on_error(tmp_path):
 previous=Path.cwd()
 with pytest.raises(RuntimeError):
  with in_output_folder(tmp_path/'maps'):raise RuntimeError('test')
 assert Path.cwd()==previous
