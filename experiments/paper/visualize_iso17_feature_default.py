"""Predeclared ISO17 example: force arrows and free trajectory, frozen C models."""

import json
import hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from train_iso17_feature_kernels import ROOT
from iso17_data import ATOMS, RAW, atomic_masses, graph, protocol, read_frames, topology_map
from taskkernel_gnn.models.molecular_features import MolecularFeatureEnergyModel


OUT=ROOT/'visualizations'
COLORS={1:'#e2e6eb',6:'#3d739b',8:'#dc554f'}
SIZES={1:40,6:95,8:110}


def axes_limits(sequences):
    xyz=np.concatenate(sequences,axis=0)
    lo,hi=xyz.min(axis=(0,1)),xyz.max(axis=(0,1))
    return (lo+hi)/2,max(float(np.max(hi-lo))*.58,1.5)


def draw(ax,xyz,bonds,numbers,center,radius,title,force=None):
    ax.clear();ax.set_title(title,fontsize=11)
    ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
           zlim=(center[2]-radius,center[2]+radius),xlabel='x (Angstrom)',ylabel='y',zlabel='z')
    ax.view_init(elev=22,azim=40)
    ax.set_box_aspect((1,1,1))
    for i,j in bonds:
        ax.plot(*xyz[[i,j]].T,color='#74808a',linewidth=1.6,alpha=.7)
    ax.scatter(*xyz.T,c=[COLORS[int(z)] for z in numbers],s=[SIZES[int(z)] for z in numbers],
               edgecolors='#33404e',linewidth=.4,depthshade=False)
    if force is not None:
        # One physical scale for both panels; no vector-by-vector normalization.
        ax.quiver(*xyz.T,*(.18*force).T,color='#7844a8',arrow_length_ratio=.28,linewidth=1.1,normalize=False)


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    start=protocol()['rollout_starts'][0]
    rows=read_frames(RAW/'test_other.db',list(range(start,start+102)),5000,101)
    truth=rows.position;reference_force=rows.force
    numbers=rows.numbers[0];mol=rows.molecules[:1];tops=topology_map(valence_cap=True)
    assert np.all(rows.numbers==numbers) and len(set(rows.molecules.tolist()))==1
    bonds=tops[int(mol[0])].bond_pairs
    mass=atomic_masses[numbers][:,None].astype(np.float32)
    forecasts,forces={},{ }
    metrics={'model':'C: original kernels + continuous filter','budget_frames':128,'steps':100,
             'reference_start_row_id':start,'selection':'first predeclared rollout start; not selected by error',
             'dt_fs':1.,'integrator':'position Verlet, first two reference frames initialize velocity',
             'scope':'one unseen-molecule window, qualitative demonstration, not aggregate rollout validation',
             'force_visualization':'reference coordinates on both panels; force magnitudes use identical scale',
             'seeds':{}}
    def predict(model,xyz,molecules):
        edges,ctx=graph(xyz.astype(np.float32),molecules,tops)
        ctx['position'].requires_grad_(True)
        _,force,_=model(edges,ctx)
        return force.detach().numpy()
    for seed in [0,42,37]:
        file=ROOT/'C'/f'budget_128_seed_{seed}'/'checkpoint.pt'
        p=torch.load(file,map_location='cpu',weights_only=False)
        model=MolecularFeatureEnergyModel(variant='C',**p['experiment_metadata']['architecture'])
        model.load_state_dict(p['model']);model.eval()
        forces[seed]=predict(model,truth,rows.molecules)
        sequence=[truth[0].copy(),truth[1].copy()]
        for step in range(100):
            f=predict(model,sequence[-1][None],mol)[0]
            following=(2*sequence[-1]-sequence[-2]+.009648533*f/mass).astype(np.float32)
            if not np.isfinite(following).all() or np.abs(following).max()>1000:
                raise RuntimeError(f'Example rollout failed: seed {seed}, step {step+1}')
            sequence.append(following)
        forecasts[seed]=np.stack(sequence)
        tri=np.triu_indices(ATOMS,1)
        pair=lambda x:np.linalg.norm(x[:,None,:]-x[None,:,:],axis=-1)[tri]
        metrics['seeds'][str(seed)]={'checkpoint_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                                   'reference_coordinate_force_mae_ev_per_ang':float(np.abs(forces[seed]-reference_force).mean()),
                                   'selected_steps':{str(s):{'position_mae_ang':float(np.abs(forecasts[seed][s+1]-truth[s+1]).mean()),
                                                           'pair_distance_mae_ang':float(np.abs(pair(forecasts[seed][s+1])-pair(truth[s+1])).mean())} for s in [1,20,50,100]}}
    oracle=[truth[0].copy(),truth[1].copy()]
    for step in range(100):
        oracle.append((2*oracle[-1]-oracle[-2]+.009648533*reference_force[step+1]/mass).astype(np.float32))
    metrics['reference_force_integration_position_mae_step100_ang']=float(np.abs(oracle[-1]-truth[-1]).mean())
    np.savez_compressed(OUT/'example_arrays.npz',reference=truth,reference_force=reference_force,numbers=numbers,
                        **{f'positions_seed_{s}':v for s,v in forecasts.items()},
                        **{f'force_seed_{s}':v for s,v in forces.items()})
    (OUT/'example_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    # Motion montage: one reference and all three seeds, identical physical camera scale.
    center,radius=axes_limits([truth,*forecasts.values()])
    fig=plt.figure(figsize=(14,13))
    for row,s in enumerate([1,20,50,100]):
        for col,(name,seq) in enumerate([('Reference',truth),*[(f'C seed {k}',v) for k,v in forecasts.items()]]):
            ax=fig.add_subplot(4,4,row*4+col+1,projection='3d')
            draw(ax,seq[s+1],bonds,numbers,center,radius,f'{name} | rollout step {s}')
    fig.suptitle('ISO17 unseen molecule: free rollout from two reference frames\nH: white, C: blue, O: red; common axes and fixed display bonds',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'rollout_three_seeds.png',dpi=140);plt.close(fig)
    # Separate single-example animation; both panels carry positions from their own sequence.
    center,radius=axes_limits([truth,forecasts[0]])
    fig=plt.figure(figsize=(11,5.5));axes=[fig.add_subplot(1,2,i,projection='3d') for i in [1,2]]
    label=fig.text(.5,.03,'',ha='center',fontsize=10)
    def motion(frame):
        draw(axes[0],truth[frame],bonds,numbers,center,radius,'ISO17 reference')
        draw(axes[1],forecasts[0][frame],bonds,numbers,center,radius,'TaskKernel-GNN C | seed 0')
        step=max(frame-1,0)
        label.set_text(f'Free rollout step {step} | position MAE {np.abs(forecasts[0][frame]-truth[frame]).mean():.3f} Angstrom\nTwo initial frames provided; subsequent model forces evaluated at predicted positions')
    displayed=[0,1,*range(3,102,2)]
    FuncAnimation(fig,motion,frames=displayed,interval=125).save(OUT/'rollout_seed0.gif',writer=PillowWriter(fps=8),dpi=90)
    motion(101);fig.savefig(OUT/'rollout_seed0_step100.png',dpi=140);plt.close(fig)
    # Force animation: shared reference positions, arrows are the predicted quantity.
    xyz=np.concatenate([truth,truth+.18*reference_force,truth+.18*forces[0]],axis=0)
    center,radius=axes_limits([xyz])
    fig=plt.figure(figsize=(11,5.5));axes=[fig.add_subplot(1,2,i,projection='3d') for i in [1,2]]
    label=fig.text(.5,.03,'',ha='center',fontsize=10)
    def arrows(frame):
        draw(axes[0],truth[frame],bonds,numbers,center,radius,'Reference forces',reference_force[frame])
        draw(axes[1],truth[frame],bonds,numbers,center,radius,'Predicted forces | C seed 0',forces[0][frame])
        label.set_text(f'Reference frame {frame} | force component MAE {np.abs(forces[0][frame]-reference_force[frame]).mean():.3f} eV/Angstrom\nSame reference coordinates; purple arrows: direction and magnitude, shared scale 0.18 Angstrom per eV/Angstrom')
    FuncAnimation(fig,arrows,frames=displayed,interval=125).save(OUT/'force_on_reference_seed0.gif',writer=PillowWriter(fps=8),dpi=90)
    arrows(50);fig.savefig(OUT/'force_on_reference_seed0_frame50.png',dpi=140);plt.close(fig)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
