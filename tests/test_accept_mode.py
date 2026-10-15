from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest,yaml
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from flare.bffs.sgp.calculator import SGP_Calculator
from mlputils import offline_training as o
ROOT=Path(__file__).resolve().parents[1]

def config():
 cfg=yaml.safe_load((ROOT/'workflows_examples/offline_setup.yaml').read_text())
 cfg.update(files_prefix='accept',nr_initial_envs=2,min_optimize=1,max_optimize=10,optimize_every=1,call_threshold=0.,add_threshold=1e-15)
 cfg['flare_calc']['descriptors'][0].update(nmax=2,lmax=1)
 cfg['flare_calc']['single_atom_energies']=[0.]
 return cfg

def frames():
 result=[]
 for shift in [0.,.1,.2]:
  a=bulk('Cu','fcc',a=3.6,cubic=True);a.positions[0,0]+=shift;a.calc=EMT()
  a.calc=SinglePointCalculator(a,energy=a.get_potential_energy(),forces=a.get_forces(),stress=a.get_stress())
  result.append(a)
 return result

@pytest.mark.parametrize('loss',['negative_likelihood','huber'])
def test_native_rollback_retains_added_environments_and_predictions(monkeypatch,loss):
 cfg=config();calc,kernels=o.get_sgp_calc(cfg['flare_calc'])
 data=frames()
 for a in data[:2]:
  e,f,s=o.get_dft_data(a);calc.gp_model.update_db(a,energy=e,forces=f,stress=s,custom_range=[0,1])
 gp=calc.gp_model.sparse_gp;old=np.array(gp.hyperparameters,copy=True)
 calc.calculate(data[-1]);expected={k:np.array(calc.results[k],copy=True) for k in ['energy','forces']}
 count=len(gp.training_structures);indices=[[int(i) for i in v] for v in gp.sparse_indices[0]]
 def optimizer(fun,x,args,**kw):
  candidate=x.copy();candidate[0]=99.
  actual=fun(candidate,*args)
  return SimpleNamespace(x=candidate,fun=actual[0] if isinstance(actual,tuple) else actual,jac=np.zeros(4))
 monkeypatch.setattr(o,'minimize',optimizer)
 options=dict(cfg['optimizer_options']);options['loss_function_config']={'name':loss,'weights':[1.,.1,0.]}
 assert o.optimize_hyps(gp,**options)
 np.testing.assert_array_equal(gp.hyperparameters,old)
 assert len(gp.training_structures)==count
 assert [[int(i) for i in v] for v in gp.sparse_indices[0]]==indices
 calc.calculate(data[-1])
 for k in expected:np.testing.assert_allclose(calc.results[k],expected[k],rtol=1e-8,atol=1e-8)

@pytest.mark.parametrize('explicit',[True,False])
def test_training_restores_last_successful_parameters_and_exports_environments(tmp_path,monkeypatch,explicit):
 monkeypatch.chdir(tmp_path);cfg=config();cfg['output_folder']='results'
 if explicit:cfg['when_rollback']='accept'
 else:cfg.pop('when_rollback',None)
 data=frames();accepted=[];calls=[]
 def optimizer(fun,x,args,**kw):
  candidate=x.copy()
  if not calls:
   candidate*=1.05;accepted.append(candidate.copy())
  else:candidate[0]=99.
  calls.append(candidate.copy());value,grad=fun(candidate,*args)
  return SimpleNamespace(x=candidate,fun=value,jac=grad)
 monkeypatch.setattr(o,'minimize',optimizer)
 o.train_offline(cfg,data,[data[-1]])
 assert len(calls)==2
 loaded,kernels=SGP_Calculator.from_file('results/accept_model.json')
 np.testing.assert_allclose(loaded.gp_model.hyps,accepted[0])
 assert len(loaded.gp_model.sparse_gp.training_structures)==3
 assert len(loaded.gp_model.sparse_gp.sparse_indices[0])==3
 assert all(len(v)>0 for v in loaded.gp_model.sparse_gp.sparse_indices[0])
 log=Path('results/training_accept.log').read_text()
 assert 'kept the newly added environments (accept mode)' in log
 logged=np.loadtxt('results/hyps_accept.dat')
 np.testing.assert_allclose(logged[0,1:],accepted[0])
 np.testing.assert_allclose(logged[1,1:],accepted[0])
 a=data[-1].copy();a.calc=loaded;assert np.isfinite(a.get_potential_energy())

def test_discard_is_rejected_before_model_initialization(monkeypatch):
 monkeypatch.setattr(o,'get_sgp_calc',lambda cfg:pytest.fail('model initialized before validation'))
 with pytest.raises(ValueError,match='discarding environments is not implemented'):
  o.train_offline({'when_rollback':'discard'},[],[])
