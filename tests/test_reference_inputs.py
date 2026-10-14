from pathlib import Path
import sys
import numpy as np
import pytest,yaml
from mlputils import reference_dft_maker as d
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('symbol',['Cu','Al','Au'])
@pytest.mark.parametrize('kpoints',['gamma',8])
def test_example_generates_all_inputs(tmp_path,monkeypatch,symbol,kpoints):
 monkeypatch.chdir(tmp_path)
 params=yaml.safe_load((ROOT/'workflows_examples/parameters_relax.yaml').read_text())
 params.update(kpts=kpoints,output_folder='output',koffset='None')
 Path('parameters.yaml').write_text(yaml.safe_dump(params))
 monkeypatch.setattr(sys,'argv',['irrelevant'])
 d.write_all_inputs(['reference_dft_maker.py','input',symbol,'parameters.yaml'])
 files=list(Path('output').glob('*.pwi'));assert len(files)==31
 for path in files:
  text=path.read_text()
  is_cluster=any(tag in path.name for tag in ['iso','dimer','Ih','Oh','Dh'])
  if kpoints=='gamma' or is_cluster:assert 'K_POINTS gamma' in text,path.name
  else:
   line=text.split('K_POINTS automatic\n')[1].splitlines()[0]
   mesh=list(map(int,line.split()));assert len(mesh)==6 and all(k>=1 for k in mesh[:3]) and mesh[3:]==[0,0,0]
 assert not list(tmp_path.glob('*.pwi'))

@pytest.mark.parametrize('kpoints',['gamma',8])
def test_phonon_inputs(tmp_path,monkeypatch,kpoints):
 monkeypatch.chdir(tmp_path)
 params=yaml.safe_load((ROOT/'workflows_examples/parameters_relax.yaml').read_text())
 params.update(kpts=kpoints,koffset=None,output_folder='phonons')
 d.input_phonons('Cu',3.572,'pp/',params,supercell_size=2)
 text=Path('phonons/Cu_phonopy_structure_0.pwi').read_text()
 assert ('K_POINTS gamma' in text)==(kpoints=='gamma')
 assert Path('phonons/Cu_phonopy.yaml').exists()

def test_relax_offset_uses_koffset(tmp_path,monkeypatch):
 monkeypatch.chdir(tmp_path)
 params=yaml.safe_load((ROOT/'workflows_examples/parameters_relax.yaml').read_text())
 params.update(kpts=8,koffset=[1,0,1],output_folder='out')
 d.input_fcc_relax('Cu',3.572,'pp/',params)
 text=Path('out/Cu_fcc_relax.pwi').read_text()
 assert list(map(int,text.split('K_POINTS automatic\n')[1].splitlines()[0].split()))==[8,8,8,1,0,1]

@pytest.mark.parametrize('args',[['script','input'],['script','input','Cu']])
def test_missing_cli_arguments(args):
 with pytest.raises(SystemExit,match='parameters.yaml'):d.write_all_inputs(args)

def test_rescaled_mesh_stays_positive_and_rounds():
 assert d.kpts_equiv(100,8,3.572)==1
 assert d.kpts_equiv(4,8,4,floor=False)==8
 assert d.kpts_equiv(3,8,4,floor=False)==11


def test_electronic_iteration_setting_and_explicit_override():
 params=yaml.safe_load((ROOT/'workflows_examples/parameters_relax.yaml').read_text())
 assert d.input_from_yaml(params,4,'pp/')['electrons']['electron_maxstep']==300
 assert d.input_from_yaml(params,1,'pp/',electron_maxstep=1500)['electrons']['electron_maxstep']==1500
