# Generate all publication figures (violin, degree heatmap, IR/IC scatter, rarity plot).

import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as ss

# ── Configure these paths before running ──────────────────────────────────────
SYMBTR_TXT_DIR = Path("/path/to/SymbTr/txt")
OUTPUT_DIR     = Path("data/idyom_output")
FIGURE_DIR     = Path("figures")

DATASET_IDS = {
    "ussak":    {"koma": 200, "tet12": 201},
    "huseyni":  {"koma": 202, "tet12": 203},
    "nihavent": {"koma": 204, "tet12": 205},
}

MAQAMS = ["ussak", "huseyni", "nihavent"]
LABELS = {"ussak": "Uşşak", "huseyni": "Hüseyni", "nihavent": "Nihavend"}
COLOR  = {"ussak": "#C94B2B", "huseyni": "#C49A00", "nihavent": "#2B6CB0"}

KOMA_LARGE, KOMA_SMALL = 27, 9
TET_LARGE,  TET_SMALL  = 6,  2

matplotlib.rcParams.update({"font.size": 10, "axes.titlesize": 10,
                             "axes.labelsize": 10})

# ── Shared data helpers ───────────────────────────────────────────────────────

def _int(s):
    try: return int(s)
    except: return None

def read_dat(p):
    rows = []
    with open(p) as f:
        hdr = f.readline().split()
        for line in f:
            pts = line.split(); row = dict(zip(hdr, pts))
            if row.get("cpitch.ic","NA") == "NA": continue
            rows.append({"mid":int(row["melody.id"]),"nid":int(row["note.id"]),
                         "cp":int(row["cpitch"]),"ic":float(row["cpitch.ic"]),
                         "name":row.get("melody.name","").strip('"')})
    return rows

def get_karar(makam):
    db = OUTPUT_DIR / makam / "koma" / "idyom.db"
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT COMPOSITION_ID,CPITCH FROM mtp_event "
                         "WHERE DATASET_ID=? ORDER BY COMPOSITION_ID,ONSET",
                         (DATASET_IDS[makam]["koma"],)).fetchall()
    last = {}
    for cid, cp in rows: last[cid] = cp
    return last

def build_records(makam):
    dat_k = sorted((OUTPUT_DIR/makam/"koma").glob("*.dat"))[0]
    dat_t = sorted((OUTPUT_DIR/makam/"tet12").glob("*.dat"))[0]
    karar = get_karar(makam)
    rk    = read_dat(dat_k); rt = read_dat(dat_t)
    idx_t = {(r["mid"],r["nid"]): r["ic"] for r in rt}
    recs  = []
    for r in rk:
        ic_t = idx_t.get((r["mid"],r["nid"]))
        if ic_t is None: continue
        k = karar.get(r["mid"]-1)
        if k is None: continue
        recs.append({"mid":r["mid"],"name":r["name"],"cp_koma":r["cp"],
                     "cp_tet":round(r["cp"]*12/53),"ic_koma":r["ic"],"ic_tet":ic_t,
                     "delta_ic":r["ic"]-ic_t,"deg":(r["cp"]-k)%53})
    return recs

def piece_dic(recs):
    per = defaultdict(list); names = {}
    for r in recs:
        per[r["mid"]].append(r["delta_ic"]); names[r["mid"]] = r["name"]
    return {(m,names[m]): np.mean(v) for m,v in per.items()}

def ir_score(I1, I2, large, small):
    a1,a2 = abs(I1),abs(I2)
    s = (I1>0 and I2>0) or (I1<0 and I2<0)
    rdir = (1.0 if s else 0.0) if a1>large else (0.0 if s else 1.0) if a1<=small else 0.0
    return rdir + (1.0 if a2<a1 else 0.0) + (1.0 if abs(I1+I2)<=small else 0.0) + \
           max(0.0,1.0-a2/large) + (1.0 if (not s) and a2<=small else 0.0)

def parse_txt(path):
    rows = []
    with open(path,encoding="utf-8") as f:
        for line in f:
            pts = line.rstrip("\n").split("\t")
            if len(pts)<9 or _int(pts[1])!=9: continue
            k = _int(pts[4])
            if not k or k<=0: continue
            rows.append((k,round(k*12/53)))
    return rows

def compute_ir(makam):
    db = OUTPUT_DIR/makam/"koma"/"idyom.db"
    with sqlite3.connect(db) as c:
        pieces = c.execute("SELECT COMPOSITION_ID,DESCRIPTION FROM mtp_composition "
                           "WHERE DATASET_ID=? ORDER BY COMPOSITION_ID",
                           (DATASET_IDS[makam]["koma"],)).fetchall()
    karar_map = get_karar(makam)
    piece_recs = []; deg_recs = defaultdict(list)
    for cid_db, name in pieces:
        txt = SYMBTR_TXT_DIR/(name+".txt")
        if not txt.exists(): continue
        notes = parse_txt(txt)
        if len(notes)<3: continue
        karar = karar_map.get(cid_db)
        if karar is None: continue
        ks=[n[0] for n in notes]; ts=[n[1] for n in notes]
        sks,sts = [],[]
        for i in range(len(notes)-2):
            sks.append(ir_score(ks[i+1]-ks[i],ks[i+2]-ks[i+1],KOMA_LARGE,KOMA_SMALL))
            sts.append(ir_score(ts[i+1]-ts[i],ts[i+2]-ts[i+1],TET_LARGE,TET_SMALL))
            deg_recs[(ks[i+2]-karar)%53].append(sks[-1]-sts[-1])
        if not sks: continue
        piece_recs.append({"name":name,"ir_koma":np.mean(sks),"ir_tet":np.mean(sts),
                           "delta_ir":np.mean(sks)-np.mean(sts),"n":len(sks)})
    return piece_recs, deg_recs

# ── Figures ───────────────────────────────────────────────────────────────────

def fig_violin(all_recs):
    piece_data = {m: list(piece_dic(all_recs[m]).values()) for m in MAQAMS}
    fig, ax = plt.subplots(figsize=(4.8,4.5))
    vp = ax.violinplot([piece_data[m] for m in MAQAMS], positions=[1,2,3],
                       showmedians=True, showextrema=False, widths=0.55)
    for body,m in zip(vp["bodies"],MAQAMS):
        body.set_facecolor(COLOR[m]); body.set_alpha(0.72); body.set_edgecolor("none")
    vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)
    rng = np.random.default_rng(42)
    for pos,m in zip([1,2,3],MAQAMS):
        jx = pos+rng.uniform(-.09,.09,len(piece_data[m]))
        ax.scatter(jx,piece_data[m],color=COLOR[m],s=11,alpha=0.45,lw=0,zorder=3)
    ax.axhline(0,color="#888",lw=0.8,ls="--")
    ax.set_xticks([1,2,3]); ax.set_xticklabels([LABELS[m] for m in MAQAMS],fontsize=11)
    ax.set_ylabel("Mean ΔIC per piece  [bits]")
    ax.set_title("Microtonal vs. 12-TET information content\n(piece level)",fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURE_DIR/"violin_piece_dic.pdf"
    fig.savefig(out,bbox_inches="tight")
    fig.savefig(FIGURE_DIR/"violin_piece_dic.png",dpi=300,bbox_inches="tight")
    plt.close(fig); print(f"  Saved {out.name}")

def fig_heatmap(all_recs):
    N_MIN = 50
    matrices = {}; all_degs = set()
    for m in MAQAMS:
        dd = defaultdict(list)
        for r in all_recs[m]: dd[r["deg"]].append(r["delta_ic"])
        matrices[m] = {d:np.mean(v) for d,v in dd.items() if len(v)>=N_MIN}
        all_degs.update(matrices[m])
    degs = sorted(all_degs)
    data = np.full((3,len(degs)),np.nan)
    for i,m in enumerate(MAQAMS):
        for j,d in enumerate(degs):
            if d in matrices[m]: data[i,j]=matrices[m][d]
    masked = np.ma.array(data,mask=np.isnan(data))
    vmax = max(abs(np.nanmin(data)),abs(np.nanmax(data)))
    cmap = matplotlib.colormaps["RdBu_r"].copy(); cmap.set_bad("#DDDDDD")
    fig,ax = plt.subplots(figsize=(max(10,len(degs)*.35),2.7))
    im = ax.imshow(masked,aspect="auto",cmap=cmap,
                   norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),interpolation="nearest")
    cb = fig.colorbar(im,ax=ax,shrink=0.85,pad=0.02); cb.set_label("Mean ΔIC  [bits]")
    ax.set_yticks(range(3)); ax.set_yticklabels([LABELS[m] for m in MAQAMS],fontsize=10)
    ax.set_xticks(range(len(degs)))
    ax.set_xticklabels([str(d) for d in degs],fontsize=7.5,rotation=45,ha="right")
    ax.set_xlabel("Tonic-relative scale degree  (koma mod 53)")
    ax.set_title(f"Degree × Maqam  ΔIC heatmap  (n ≥ {N_MIN})",fontsize=10)
    for j,d in enumerate(degs):
        if d==8:
            ax.add_patch(mpatches.Rectangle((j-.5,-.5),1,3,fill=False,edgecolor="black",lw=1.8))
            ax.text(j,3.15,"d=8",ha="center",va="bottom",fontsize=7.5,fontweight="bold")
        elif d==7:
            ax.add_patch(mpatches.Rectangle((j-.5,-.5),1,3,fill=False,edgecolor="#444",lw=1,ls="--"))
            ax.text(j,3.15,"d=7",ha="center",va="bottom",fontsize=7.5,color="#444")
    fig.tight_layout()
    out = FIGURE_DIR/"heatmap_degree.pdf"
    fig.savefig(out,bbox_inches="tight")
    fig.savefig(FIGURE_DIR/"heatmap_degree.png",dpi=300,bbox_inches="tight")
    plt.close(fig); print(f"  Saved {out.name}")

def fig_scatter_ir_ic(all_recs):
    dic_name = {}
    for m in MAQAMS:
        for (mid,name),v in piece_dic(all_recs[m]).items():
            dic_name[(m,name)] = v
    fig,ax = plt.subplots(figsize=(5,4.5))
    all_ics,all_irs = [],[]
    for m in MAQAMS:
        recs,_ = compute_ir(m)
        dir_name = {r["name"]:r["delta_ir"] for r in recs}
        paired = [(dic_name[(m,n)],dir_name[n]) for n in dir_name if (m,n) in dic_name]
        if not paired: continue
        ics,irs = zip(*paired)
        ax.scatter(ics,irs,color=COLOR[m],alpha=0.45,s=18,label=LABELS[m],zorder=3)
        all_ics.extend(ics); all_irs.extend(irs)
    slope,inter,r,p,_ = ss.linregress(all_ics,all_irs)
    xl = np.linspace(min(all_ics)-.01,max(all_ics)+.01,100)
    ax.plot(xl,slope*xl+inter,"k-",lw=1.0,alpha=0.6,label=f"Overall: r={r:.2f}, p={p:.4f}")
    ax.axhline(0,color="#888",lw=0.6,ls="--"); ax.axvline(0,color="#888",lw=0.6,ls="--")
    ax.set_xlabel("ΔIC  [bits]   (koma − 12-TET statistical surprise)")
    ax.set_ylabel("ΔIR   (koma − 12-TET structural expectedness)")
    ax.set_title("IDyOM vs. IR: complementary expectation models\n(piece level)",fontsize=10)
    ax.legend(fontsize=8.5,framealpha=0.85); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURE_DIR/"scatter_ir_ic.pdf"
    fig.savefig(out,bbox_inches="tight")
    fig.savefig(FIGURE_DIR/"scatter_ir_ic.png",dpi=300,bbox_inches="tight")
    plt.close(fig); print(f"  Saved {out.name}")

def fig_rarity(all_recs):
    fig,axes = plt.subplots(1,3,figsize=(13,4.5))
    fig.suptitle("ΔIC vs. degree frequency  (n ≥ 20 per degree)",fontsize=11)
    for ax,m in zip(axes,MAQAMS):
        total = len(all_recs[m]); dd = defaultdict(list)
        for r in all_recs[m]: dd[r["deg"]].append(r["delta_ic"])
        degs=[d for d,v in dd.items() if len(v)>=20]
        freqs=[len(dd[d])/total for d in degs]; deltas=[np.mean(dd[d]) for d in degs]
        colors=["#D62728" if d==8 else COLOR[m] for d in degs]
        sizes=[70 if d==8 else 28 for d in degs]
        for x,y,c,s in zip(freqs,deltas,colors,sizes): ax.scatter(x,y,color=c,s=s,alpha=0.8)
        for d,f,v in zip(degs,freqs,deltas):
            if d in (0,7,8,13,22,31,44) or abs(v)>0.5:
                ax.annotate(str(d),(f,v),fontsize=6.5,xytext=(2,2),textcoords="offset points")
        if len(freqs)>2:
            lf=np.log(freqs); sl,ic,r,p,_=ss.linregress(lf,deltas)
            xr=np.linspace(min(lf),max(lf),100)
            ax.plot(np.exp(xr),sl*xr+ic,"k--",lw=1,alpha=0.6,label=f"r={r:.2f}, p={p:.3f}")
            ax.legend(fontsize=8)
        ax.axhline(0,color="grey",lw=0.7,ls="--"); ax.set_xscale("log")
        ax.set_xlabel("Degree frequency  (log scale)")
        if m=="ussak": ax.set_ylabel("Mean ΔIC  [bits]")
        ax.set_title(LABELS[m]); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURE_DIR/"rarity.png"
    fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig); print(f"  Saved {out.name}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    all_recs = {m: build_records(m) for m in MAQAMS}
    fig_violin(all_recs)
    fig_heatmap(all_recs)
    fig_scatter_ir_ic(all_recs)
    fig_rarity(all_recs)

if __name__ == "__main__":
    main()
