#!/usr/bin/env python3
"""A5P5 - Crash-consistent credential rotation: GRU sequence classifier over rotation event sequences.

Runs everything: preprocessing (imputation, winsorisation), the proposed context-initialised GRU with an
auxiliary final-state head, three baselines (fixed-order rotation rules, logistic regression, MLP),
an ablation study, a hidden-size sensitivity study, stratified 5-fold CV with mean +- std, paired tests,
the Graphviz methodology flowchart and all figures.  Outputs go to ../results and ../figures.
"""
import os, json, time, subprocess, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (f1_score, average_precision_score, roc_auc_score, brier_score_loss,
                             precision_recall_curve, precision_score, recall_score, accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "A5P5_credential_rotations.csv")
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SEED = 2026
N_FOLDS = 5

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 100})
COL = {"Fixed-order rules": "#9e9e9e", "Logistic regression": "#6d8fb3", "MLP": "#7fa77a", "GRU-CR (proposed)": "#c0392b"}

# ----------------------------------------------------------------------------- data
TARGET = "inconsistent_state"
LEAK = ["final_state", "auth_failure_minutes", "failed_transactions", "missed_blocks"]  # post-event, never features
NETS = ["permissioned_energy_ledger", "settlement_sidechain", "public_l2_bridge"]
CTYPES = ["validator_signing_key", "rpc_tls_certificate", "node_db_password", "api_client_token", "p2p_node_key",
          "hsm_wrapping_key"]
PROTOS = ["naive_overwrite", "two_phase_commit", "dual_credential_overlap", "versioned_secret_lease"]
CRASH = ["none", "after_new_cred_created", "after_secret_store_write", "after_db_or_node_update",
         "during_client_reload", "before_old_cred_revoked"]
RECOV = ["manual", "restart_from_scratch", "journal_replay"]
STATES = ["consistent", "store_ahead_of_target", "target_ahead_of_store", "clients_split_brain", "double_sign_risk",
          "orphaned_credential"]
STEPS = ["CREATE", "STORE_WRITE", "TARGET_UPDATE", "CLIENT_RELOAD", "REVOKE_OLD", "RECOVER"]
# index of the last step that completed before the crash (-1 = none completed), and the step that crashed
CRASH_AFTER = {"none": 4, "after_new_cred_created": 0, "after_secret_store_write": 1, "after_db_or_node_update": 2,
               "during_client_reload": 2, "before_old_cred_revoked": 3}
CRASH_AT = {"none": None, "after_new_cred_created": 1, "after_secret_store_write": 2, "after_db_or_node_update": 3,
            "during_client_reload": 3, "before_old_cred_revoked": 4}


def load():
    df = pd.read_csv(DATA)
    return df


class Prep:
    """Fold-wise preprocessing: median / mode imputation with missing flags, log transform, winsorisation."""

    def fit(self, d):
        self.lag_med = np.nanmedian(np.log1p(d.secret_store_replication_lag_s.clip(lower=0)))
        self.tx_med = np.nanmedian(np.log1p(d.tx_rate_per_s.clip(lower=0)))
        self.ttl_mode = d.client_cache_ttl_s.mode().iloc[0]
        ll = np.log1p(d.secret_store_replication_lag_s.clip(lower=0).dropna())
        lt = np.log1p(d.tx_rate_per_s.clip(lower=0).dropna())
        self.lag_q = (np.percentile(ll, 1), np.percentile(ll, 99))
        self.tx_q = (np.percentile(lt, 1), np.percentile(lt, 99))
        return self

    def transform(self, d):
        o = pd.DataFrame(index=d.index)
        o["lag_miss"] = d.secret_store_replication_lag_s.isna().astype(float)
        o["tx_miss"] = d.tx_rate_per_s.isna().astype(float)
        o["ttl_miss"] = d.client_cache_ttl_s.isna().astype(float)
        o["log_lag"] = np.log1p(d.secret_store_replication_lag_s.clip(lower=0)).fillna(self.lag_med).clip(*self.lag_q)
        o["log_tx"] = np.log1p(d.tx_rate_per_s.clip(lower=0)).fillna(self.tx_med).clip(*self.tx_q)
        ttl = d.client_cache_ttl_s.fillna(self.ttl_mode)
        o["log_ttl"] = np.log1p(ttl) / np.log1p(3600)
        o["log_grace"] = np.log1p(d.old_credential_grace_s) / np.log1p(3600)
        o["ttl_gt_grace"] = (ttl > d.old_credential_grace_s).astype(float)
        o["log_clients"] = np.log1p(d.n_dependent_clients)
        o["log_nodes"] = np.log1p(d.n_ledger_nodes)
        o["wal"] = d.write_ahead_journal.astype(float)
        o["hot"] = d.client_hot_reload.astype(float)
        o["stake"] = d.validator_stake_share.astype(float)
        return o


def onehot(series, cats):
    return np.stack([(series.values == c).astype(float) for c in cats], 1)


def tabular(d, p):
    """Flat feature vector used by LR and MLP (same information as the GRU receives)."""
    o = p.transform(d)
    X = np.hstack([onehot(d.ledger_network, NETS), onehot(d.credential_type, CTYPES), onehot(d.rotation_protocol, PROTOS),
                   onehot(d.injected_crash_point, CRASH), onehot(d.recovery_mode, RECOV), o.values])
    return X


def sequences(d, p):
    """Encode every rotation run as an ordered 6-step event sequence + a static context vector.
    Token features: step one-hot(6) | status one-hot(completed, crashed-here, not-reached, recovery-op, no-op)(5)
                    | WAL flag | step-specific signals (lag, nodes, ttl, hot-reload, clients, ttl>grace, grace,
                    recovery one-hot(3)) | partial flag (crash interrupted the step mid-way).
    The partial flag is required so that "after_db_or_node_update" (crash before CLIENT_RELOAD starts) and
    "during_client_reload" (crash part-way through CLIENT_RELOAD) do not map to identical sequences."""
    o = p.transform(d)
    n = len(d)
    F = 6 + 5 + 1 + 7 + 3 + 1
    S = np.zeros((n, 6, F), dtype=np.float32)
    crash = d.injected_crash_point.values
    rec = onehot(d.recovery_mode, RECOV)
    wal = o.wal.values
    for t in range(6):
        S[:, t, t] = 1.0
        S[:, t, 11] = wal
    after = np.array([CRASH_AFTER[c] for c in crash])
    at = np.array([-1 if CRASH_AT[c] is None else CRASH_AT[c] for c in crash])
    crashed = crash != "none"
    for t in range(5):
        S[:, t, 6] = (t <= after).astype(np.float32)                    # completed
        S[:, t, 7] = (t == at).astype(np.float32)                       # crashed at this step
        S[:, t, 8] = ((t > after) & (t != at)).astype(np.float32)        # not reached before crash
    S[:, 5, 9] = crashed.astype(np.float32)                              # recovery operation runs
    S[:, 5, 10] = (~crashed).astype(np.float32)                          # no-op
    # step-specific signals
    S[:, 1, 12] = o.log_lag.values
    S[:, 2, 13] = o.log_nodes.values / np.log1p(31)
    S[:, 3, 14] = o.log_ttl.values
    S[:, 3, 15] = o.hot.values
    S[:, 3, 16] = o.log_clients.values / 5.0
    S[:, 3, 17] = o.ttl_gt_grace.values
    S[:, 4, 18] = o.log_grace.values
    S[:, 5, 19:22] = rec * crashed[:, None]
    S[:, 3, 22] = (crash == "during_client_reload").astype(np.float32)   # reload interrupted mid-way
    C = np.hstack([onehot(d.ledger_network, NETS), onehot(d.credential_type, CTYPES), onehot(d.rotation_protocol, PROTOS),
                   o[["log_clients", "log_nodes", "log_tx", "stake", "lag_miss", "tx_miss", "ttl_miss"]].values]
                  ).astype(np.float32)
    return S, C


# ----------------------------------------------------------------------------- models
class GRUCR(nn.Module):
    """Context-initialised GRU over rotation event sequences with a binary and an auxiliary state head."""

    def __init__(self, f_tok, f_ctx, hidden=48, emb=32, ctx_init=True, pool="gru"):
        super().__init__()
        self.ctx_init, self.pool = ctx_init, pool
        self.emb = nn.Sequential(nn.Linear(f_tok, emb), nn.ReLU())
        self.ctx = nn.Sequential(nn.Linear(f_ctx, hidden), nn.Tanh())
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + f_ctx, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))
        self.aux = nn.Linear(hidden, len(STATES))
        self.mean_proj = nn.Linear(emb, hidden)

    def forward(self, s, c):
        e = self.emb(s)
        if self.pool == "mean":                      # order-free ablation (bag of events)
            h = torch.tanh(self.mean_proj(e.mean(1)) + (self.ctx(c) if self.ctx_init else 0))
        else:
            h0 = self.ctx(c).unsqueeze(0) if self.ctx_init else None
            _, hn = self.gru(e, h0)
            h = hn[-1]
        return self.head(torch.cat([h, c], 1)).squeeze(1), self.aux(h)


def train_gru(S, C, y, ys, Sv, Cv, yv, seed, hidden=48, lam=0.3, ctx_init=True, pool="gru", epochs=40,
              lr=3e-3, bs=256, patience=8, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    m = GRUCR(S.shape[2], C.shape[1], hidden=hidden, ctx_init=ctx_init, pool=pool)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    St, Ct, yt, yst = map(torch.tensor, (S, C, y.astype(np.float32), ys))
    Svt, Cvt = torch.tensor(Sv), torch.tensor(Cv)
    best, best_state, wait = -1, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(yt), generator=g); tl = 0.0
        for i in range(0, len(yt), bs):
            idx = perm[i:i + bs]
            lo, la = m(St[idx], Ct[idx])
            loss = bce(lo, yt[idx]) + lam * ce(la, yst[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(Svt, Cvt)[0]).numpy()
        ap = average_precision_score(yv, pv)
        if log is not None:
            log.append({"epoch": ep + 1, "train_loss": tl / len(yt), "val_pr_auc": ap,
                        "val_logloss": float(-np.mean(yv * np.log(pv + 1e-7) + (1 - yv) * np.log(1 - pv + 1e-7)))})
        if ap > best + 1e-4:
            best, wait = ap, 0; best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(best_state); m.eval()
    return m


def gru_predict(m, S, C):
    with torch.no_grad():
        lo, la = m(torch.tensor(S), torch.tensor(C))
    return torch.sigmoid(lo).numpy(), torch.softmax(la, 1).numpy()


def rule_score(d, p):
    """Fixed-order rotation checklist: walk the orchestrator steps in order and add one point per violated
    crash-consistency rule (expert rules, not learned)."""
    o = p.transform(d)
    crashed = (d.injected_crash_point != "none").values
    s = np.zeros(len(d))
    s += 2 * (d.rotation_protocol == "naive_overwrite").values + 1 * (d.rotation_protocol == "two_phase_commit").values
    s += 1 * d.injected_crash_point.isin(["after_secret_store_write", "after_db_or_node_update",
                                          "during_client_reload"]).values
    s += 1 * (o.wal.values == 0)
    s += 1 * (d.recovery_mode == "manual").values - 1 * (d.recovery_mode == "journal_replay").values
    s += 1 * o.ttl_gt_grace.values * (d.rotation_protocol != "naive_overwrite").values
    s = np.where(crashed, s + 1, 0) + 0.1 * (d.credential_type == "validator_signing_key").values \
        + 0.1 * (d.ledger_network == "public_l2_bridge").values
    return s


def generator_oracle(df):
    """Bayes reference from the data-generating equation of gen_author5.py (paper5), see main()."""
    pr = df.rotation_protocol.values; cr = df.injected_crash_point.values
    step_risk = {
        "naive_overwrite": {"none": -2.2, "after_new_cred_created": -0.8, "after_secret_store_write": 2.8,
                            "after_db_or_node_update": 0.4, "during_client_reload": 2.2, "before_old_cred_revoked": -0.6},
        "two_phase_commit": {"none": -3.0, "after_new_cred_created": -1.6, "after_secret_store_write": -0.6,
                             "after_db_or_node_update": 2.6, "during_client_reload": 0.2, "before_old_cred_revoked": -1.2},
        "dual_credential_overlap": {"none": -3.2, "after_new_cred_created": -2.0, "after_secret_store_write": -1.2,
                                    "after_db_or_node_update": -1.4, "during_client_reload": -1.2, "before_old_cred_revoked": -0.8},
        "versioned_secret_lease": {"none": -3.4, "after_new_cred_created": -2.0, "after_secret_store_write": -1.4,
                                   "after_db_or_node_update": -1.0, "during_client_reload": 1.2, "before_old_cred_revoked": -1.0}}
    crashed = cr != "none"
    wal = df.write_ahead_journal.values; rec = df.recovery_mode.values; grace = df.old_credential_grace_s.values
    ttl = df.client_cache_ttl_s.fillna(df.client_cache_ttl_s.mode().iloc[0]).values
    lag = df.secret_store_replication_lag_s.fillna(df.secret_store_replication_lag_s.median()).clip(lower=0).values
    ncl = df.n_dependent_clients.values; ct = df.credential_type.values
    z = np.array([step_risk[a][b] for a, b in zip(pr, cr)])
    z += crashed * np.where(wal == 1, np.where(rec == "journal_replay", -2.4, -0.6), np.where(rec == "journal_replay", 0.3, 0.0))
    z += crashed * (rec == "manual") * np.where(grace < 300, 1.3, 0.3)
    z += crashed * (rec == "restart_from_scratch") * (cr == "after_db_or_node_update") * np.isin(pr, ["two_phase_commit", "dual_credential_overlap"]) * 1.0
    z += (pr == "dual_credential_overlap") * (cr == "before_old_cred_revoked") * (ttl > grace) * 2.6
    z += (ct == "validator_signing_key") * np.isin(cr, ["after_db_or_node_update", "during_client_reload"]) * (rec != "journal_replay") * 1.6
    z += np.isin(cr, ["after_secret_store_write", "during_client_reload"]) * 1.3 * (lag > 5)
    z += (~crashed) * (lag > 5) * (ncl > 50) * 1.5
    z += (0.2 * np.log1p(ncl) - 0.4 * df.client_hot_reload.values + 0.6 * (df.ledger_network.values == "public_l2_bridge")
          + 0.3 * (ct == "validator_signing_key") + 0.15 * np.log1p(lag))
    gx, gw = np.polynomial.hermite_e.hermegauss(30); gw = gw / gw.sum()
    def prob(m):
        out = np.zeros(len(z))
        for pin, wp in ((0.0, 0.96), (2.2, 0.04)):
            for x, w in zip(gx, gw):
                out += wp * w / (1 + np.exp(-(2.8 * (z + pin + 0.55 * x) + m)))
        return out
    lo, hi = -30.0, 30.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (lo, m) if prob(m).mean() > 0.329 else (m, hi)
    return prob((lo + hi) / 2)


def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr * rc / (pr + rc + 1e-12)
    i = int(np.nanargmax(f[:-1])) if len(th) else 0
    return th[i] if len(th) else 0.5


def metrics(y, p, thr, fail_tx):
    yh = (p >= thr).astype(int)
    k = int(0.25 * len(y))
    top = np.argsort(-p)[:k]
    return {"F1": f1_score(y, yh), "Precision": precision_score(y, yh, zero_division=0),
            "Recall": recall_score(y, yh), "Accuracy": accuracy_score(y, yh),
            "PR_AUC": average_precision_score(y, p), "ROC_AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, np.clip(p, 0, 1)),
            "FailTx_capture_top25": fail_tx[top].sum() / max(fail_tx.sum(), 1)}


# ----------------------------------------------------------------------------- flowchart
def flowchart():
    dot = r'''digraph G {
graph [rankdir=TB, fontname="Helvetica", fontsize=18, nodesep=0.18, ranksep=0.24, pad=0.05, compound=true, newrank=true];
node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=17, fillcolor="#eeeeee", color="#666666", penwidth=1.2, margin="0.07,0.03"];
edge  [color="#444444", arrowsize=0.8, penwidth=1.3, fontname="Helvetica", fontsize=15];

subgraph cluster_data { label="1. Data"; style="rounded,filled"; fillcolor="#f4f4f4"; color="#bbbbbb";
  raw  [label="13,000\nrotation\nruns; post-\nevent cols\ndropped"];
}
subgraph cluster_prep { label="2. Pre-processing (per fold)"; style="rounded,filled"; fillcolor="#f1f5fa"; color="#b6c6da";
  win  [label="impute, flag,\nlog, winsorise\n(P1-P99)"];
  enc  [label="event tokens x_t\n[step, status,\nWAL, signals] (1)", fillcolor="#fde3df", color="#c0392b"];
  ctxv [label="static context c\n(network, cred.,\nprotocol, load)", fillcolor="#fde3df", color="#c0392b"];
  { rank=same; enc; ctxv; }
}
subgraph cluster_model { label="3. Proposed GRU-CR"; style="rounded,filled"; fillcolor="#fdf1ef"; color="#e3a79e";
  h0   [label="h0 = tanh(W_c c)\nEq. (2)", fillcolor="#f8c9c1", color="#c0392b"];
  seq  [label="CREATE > STORE >\nTARGET > RELOAD >\nREVOKE > RECOVER", fillcolor="#f8c9c1", color="#c0392b"];
  gru  [label="GRU, t = 1..6\nh_t = f(x_t, h_t-1)\nEq. (3)-(4)", fillcolor="#f8c9c1", color="#c0392b"];
  { rank=same; h0; seq; }
  bin  [label="risk head\np, Eq. (5)", fillcolor="#f8c9c1", color="#c0392b"];
  aux  [label="state head\nq, Eq. (6)", fillcolor="#f8c9c1", color="#c0392b"];
  { rank=same; bin; gru; aux; }
  loss [label="L = BCE +\nlambda CE\nEq. (7), Adam", fillcolor="#f8c9c1", color="#c0392b"];
  stop [shape=diamond, label="val PR-AUC\nimproved in\n8 epochs?", fillcolor="#ffffff", color="#c0392b", fontsize=17, margin="0.0,0.0"];
  { rank=same; loss; stop; }
}
subgraph cluster_eval { label="4. Evaluation (5-fold CV)"; style="rounded,filled"; fillcolor="#f2f7f0"; color="#b5cdae";
  base [label="baselines:\nrules, LR, MLP"];
  thr  [label="F1-optimal\nthreshold"];
  met  [label="F1, PR-AUC, Brier,\nC25 (8), ablation,\nt-tests"];
  { rank=same; base; met; thr; }
}
{ rank=same; raw; win; }
raw -> win;
win -> enc; win -> ctxv;
enc -> seq; ctxv -> h0;
h0 -> gru [label=" init"]; seq -> gru [label=" x_t"];
bin -> gru [style=invis]; gru -> bin [constraint=false]; gru -> aux;
bin -> loss; aux -> stop [style=invis]; aux -> loss;
loss -> stop;
stop:nw -> gru:s [label="yes", fontcolor="#444444", style=dashed, constraint=false];
stop -> thr [label=" no: best\n weights"];
base -> met; met -> thr [dir=back];
}'''
    path = os.path.join(FIG, "fig1_flowchart.dot")
    open(path, "w").write(dot)
    subprocess.run(["dot", "-Tpng", "-Gdpi=300", path, "-o", os.path.join(FIG, "fig1_flowchart.png")], check=True)


def write_setup_table():
    npar = sum(x.numel() for x in GRUCR(23, 20, hidden=48).parameters())
    rows = [("GRU-CR", f"embed 32, H=48, head 32, dropout 0.1, {npar:,} params"),
            ("Optimiser", "Adam, lr 3e-3, weight decay 1e-4, batch 256"),
            ("Epochs / patience", "max 40 / 8 on validation PR-AUC"), ("Auxiliary weight lambda", "0.3"),
            ("Logistic regression", "L2, C=1, lbfgs, max_iter 3000"),
            ("MLP", "(64, 32), alpha=1e-3, lr 1e-3, early stopping (15%, 10 iter)"),
            ("Fixed-order rules", "integer checklist score; cut-off tuned on inner validation"),
            ("Protocol", "stratified 5-fold CV, inner 15% validation"), ("Seeds", "2026 + fold index"),
            ("Software", "PyTorch (CPU, 1 thread), scikit-learn")]
    pd.DataFrame(rows, columns=["Setting", "Value"]).to_csv(os.path.join(RES, "table2_hyperparameters.csv"), index=False)


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    flowchart()
    write_setup_table()
    df = load()
    y_all = df[TARGET].values
    ys_all = df.final_state.map({s: i for i, s in enumerate(STATES)}).values
    ftx_all = df.failed_transactions.values.astype(float)

    # dataset summary
    summ = [("Rotation runs", f"{len(df):,}"), ("Inconsistent runs", f"{y_all.sum():,} ({100*y_all.mean():.1f}%)"),
            ("Runs with injected crash", f"{(df.injected_crash_point!='none').sum():,} ({100*(df.injected_crash_point!='none').mean():.1f}%)"),
            ("Missing: cache TTL / repl. lag / tx rate", f"{100*df.client_cache_ttl_s.isna().mean():.1f}% / {100*df.secret_store_replication_lag_s.isna().mean():.1f}% / {100*df.tx_rate_per_s.isna().mean():.1f}%"),
            ("Median dependent clients (IQR)", f"{df.n_dependent_clients.median():.0f} ({df.n_dependent_clients.quantile(.25):.0f}-{df.n_dependent_clients.quantile(.75):.0f})"),
            ("Median replication lag, s", f"{df.secret_store_replication_lag_s.median():.2f}"),
            ("Total failed transactions", f"{int(ftx_all.sum()):,}"),
            ("Share of failed tx in inconsistent runs", f"{100*ftx_all[y_all==1].sum()/ftx_all.sum():.1f}%")]
    pd.DataFrame(summ, columns=["Item", "Value"]).to_csv(os.path.join(RES, "table1_dataset_summary.csv"), index=False)
    pc = df.pivot_table(index="rotation_protocol", columns="injected_crash_point", values=TARGET, aggfunc="mean")
    pc = pc.loc[PROTOS, CRASH]
    pc.to_csv(os.path.join(RES, "protocol_crash_inconsistency_rate.csv"))

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    rows, abl_rows, sens_rows = [], [], []
    curves = {}
    oof = {k: np.zeros(len(df)) for k in COL}
    oof_aux = np.zeros((len(df), len(STATES)))
    logs = []
    ablations = {"Full GRU-CR": dict(), "w/o context-initialised h0": dict(ctx_init=False),
                 "w/o auxiliary state head (lambda=0)": dict(lam=0.0), "Order-free (mean-pooled events)": dict(pool="mean")}
    hidden_grid = [8, 16, 32, 48, 96]
    timing = {k: [] for k in COL}
    for fold, (tr, te) in enumerate(skf.split(df, y_all)):
        dtr_full, dte = df.iloc[tr], df.iloc[te]
        itr, iva = train_test_split(np.arange(len(tr)), test_size=0.15, stratify=y_all[tr], random_state=SEED + fold)
        dtr, dva = dtr_full.iloc[itr], dtr_full.iloc[iva]
        p = Prep().fit(dtr)
        ytr, yva, yte = dtr[TARGET].values, dva[TARGET].values, dte[TARGET].values
        ftx = dte.failed_transactions.values.astype(float)
        preds = {}
        # rules
        ts = time.time()
        sva = rule_score(dva, p); thr = best_threshold(yva, sva); ste = rule_score(dte, p)
        pr_te = (ste - ste.min()) / (ste.max() - ste.min() + 1e-9)
        thr_n = (thr - ste.min()) / (ste.max() - ste.min() + 1e-9)
        preds["Fixed-order rules"] = (pr_te, thr_n); timing["Fixed-order rules"].append(time.time() - ts)
        # LR
        ts = time.time()
        Xtr, Xva, Xte = tabular(dtr, p), tabular(dva, p), tabular(dte, p)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, ytr)
        preds["Logistic regression"] = (lr.predict_proba(Xte)[:, 1], best_threshold(yva, lr.predict_proba(Xva)[:, 1]))
        timing["Logistic regression"].append(time.time() - ts)
        # MLP
        ts = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=200,
                            early_stopping=True, validation_fraction=0.15, n_iter_no_change=10,
                            random_state=SEED + fold).fit(Xtr, ytr)
        preds["MLP"] = (mlp.predict_proba(Xte)[:, 1], best_threshold(yva, mlp.predict_proba(Xva)[:, 1]))
        timing["MLP"].append(time.time() - ts)
        # GRU-CR
        Str, Ctr = sequences(dtr, p); Sva, Cva = sequences(dva, p); Ste, Cte = sequences(dte, p)
        ystr = dtr.final_state.map({s: i for i, s in enumerate(STATES)}).values
        ts = time.time()
        lg = []
        m = train_gru(Str, Ctr, ytr, ystr, Sva, Cva, yva, seed=SEED + fold, log=lg)
        for r in lg: r["fold"] = fold + 1
        logs += lg
        pva, _ = gru_predict(m, Sva, Cva); pte, ate = gru_predict(m, Ste, Cte)
        preds["GRU-CR (proposed)"] = (pte, best_threshold(yva, pva))
        timing["GRU-CR (proposed)"].append(time.time() - ts)
        oof_aux[te] = ate
        for k, (pp, th) in preds.items():
            r = metrics(yte, pp, th, ftx); r.update(method=k, fold=fold + 1); rows.append(r)
            oof[k][te] = pp
        # ablation
        for name, kw in ablations.items():
            if name == "Full GRU-CR":
                pv2, pt2 = pva, pte
            else:
                ma = train_gru(Str, Ctr, ytr, ystr, Sva, Cva, yva, seed=SEED + fold, **kw)
                pv2, _ = gru_predict(ma, Sva, Cva); pt2, _ = gru_predict(ma, Ste, Cte)
            r = metrics(yte, pt2, best_threshold(yva, pv2), ftx); r.update(variant=name, fold=fold + 1); abl_rows.append(r)
        # sensitivity on hidden size
        for h in hidden_grid:
            if h == 48:
                pv2, pt2 = pva, pte
            else:
                mh = train_gru(Str, Ctr, ytr, ystr, Sva, Cva, yva, seed=SEED + fold, hidden=h)
                pv2, _ = gru_predict(mh, Sva, Cva); pt2, _ = gru_predict(mh, Ste, Cte)
            r = metrics(yte, pt2, best_threshold(yva, pv2), ftx); r.update(hidden=h, fold=fold + 1)
            r["params"] = sum(x.numel() for x in GRUCR(Str.shape[2], Ctr.shape[1], hidden=h).parameters())
            sens_rows.append(r)
        print(f"fold {fold+1} done  {time.time()-t0:.0f}s", flush=True)

    R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "cv_fold_metrics.csv"), index=False)
    A = pd.DataFrame(abl_rows); A.to_csv(os.path.join(RES, "ablation_fold_metrics.csv"), index=False)
    SN = pd.DataFrame(sens_rows); SN.to_csv(os.path.join(RES, "sensitivity_hidden_fold_metrics.csv"), index=False)
    pd.DataFrame(logs).to_csv(os.path.join(RES, "gru_training_log.csv"), index=False)
    mets = ["F1", "Precision", "Recall", "PR_AUC", "ROC_AUC", "Brier", "FailTx_capture_top25"]

    def summarise(D, key, order):
        out = []
        for k in order:
            s = D[D[key] == k]; r = {key: k}
            for mt in mets:
                r[mt + "_mean"] = s[mt].mean(); r[mt + "_std"] = s[mt].std(ddof=1)
            out.append(r)
        return pd.DataFrame(out)
    T3 = summarise(R, "method", list(COL)); T3["train_time_s_mean"] = [np.mean(timing[k]) for k in COL]
    T3.to_csv(os.path.join(RES, "table3_main_comparison.csv"), index=False)
    T4 = summarise(A, "variant", list(ablations)); T4.to_csv(os.path.join(RES, "table4_ablation.csv"), index=False)
    T5 = summarise(SN, "hidden", hidden_grid); T5["params"] = [SN[SN.hidden == h].params.iloc[0] for h in hidden_grid]
    T5.to_csv(os.path.join(RES, "sensitivity_hidden_summary.csv"), index=False)

    # paired t-tests GRU vs each baseline
    tests = {}
    g = R[R.method == "GRU-CR (proposed)"].sort_values("fold")
    for k in list(COL)[:-1]:
        b = R[R.method == k].sort_values("fold")
        for mt in ["F1", "PR_AUC"]:
            t, pv = stats.ttest_rel(g[mt].values, b[mt].values)
            tests[f"{k}|{mt}"] = {"mean_diff": float((g[mt].values - b[mt].values).mean()), "t": float(t), "p": float(pv)}
    for v in list(ablations)[1:]:
        b = A[A.variant == v].sort_values("fold"); f = A[A.variant == "Full GRU-CR"].sort_values("fold")
        for mt in ["F1", "PR_AUC"]:
            t, pv = stats.ttest_rel(f[mt].values, b[mt].values)
            tests[f"ablation:{v}|{mt}"] = {"mean_diff": float((f[mt].values - b[mt].values).mean()), "t": float(t), "p": float(pv)}

    # per-crash-point PR-AUC / F1 analysis (out-of-fold)
    per = []
    thr_oof = {}
    for k in COL:
        # use median of fold thresholds is not stored; instead evaluate per-crash-point ranking quality and recall@global
        pass
    for c in CRASH[1:]:
        mask = (df.injected_crash_point == c).values
        r = {"crash_point": c, "n": int(mask.sum()), "inconsistent_rate": float(y_all[mask].mean())}
        for k in COL:
            r[k] = average_precision_score(y_all[mask], oof[k][mask])
        per.append(r)
    PER = pd.DataFrame(per); PER.to_csv(os.path.join(RES, "per_crash_point_pr_auc.csv"), index=False)

    # auxiliary head final_state macro-F1 on inconsistent runs (out-of-fold)
    inc = y_all == 1
    pred_state = oof_aux[:, 1:].argmax(1) + 1          # inconsistency type, conditional on an inconsistent run
    aux_macro_f1 = f1_score(ys_all[inc], pred_state[inc], average="macro")
    aux_acc_all = accuracy_score(ys_all[inc], pred_state[inc])

    # reference ceiling: generator-oracle probability from the published data-generating equation of the NEW
    # generator (gen_author5.py, paper5). It is applied to the observed (noisy, imputed) columns and marginalises
    # the two quantities a model can never see: the latent pinned-client indicator (P=0.04, +2.2 logits) and the
    # N(0, 0.55) unmodelled variation (Gauss-Hermite quadrature). The calibration offset is re-solved so that the
    # mean probability equals the generator's target prevalence 0.329. Not a model; uses knowledge of the generator.
    oracle_p = generator_oracle(df)
    p_or = oracle_p
    oracle = {"PR_AUC": float(average_precision_score(y_all, p_or)), "ROC_AUC": float(roc_auc_score(y_all, p_or)),
              "Brier": float(brier_score_loss(y_all, p_or)),
              "F1_at_best_threshold": float(max(f1_score(y_all, (p_or >= t).astype(int)) for t in np.linspace(0.1, 0.9, 81))),
              "note": "new generator equation on observed columns, latent pinned flag and N(0,0.55) noise marginalised"}

    # counterfactual protocol risk map from GRU (average predicted risk when every crashed run is re-encoded
    # with each protocol) -- uses the fold-5 model would be biased; use a model trained on all data
    p_all = Prep().fit(df)
    Sall, Call = sequences(df, p_all)
    itr, iva = train_test_split(np.arange(len(df)), test_size=0.15, stratify=y_all, random_state=SEED)
    mfull = train_gru(Sall[itr], Call[itr], y_all[itr], ys_all[itr], Sall[iva], Call[iva], y_all[iva], seed=SEED)
    cf = np.zeros((len(PROTOS), len(CRASH)))
    for i, pr in enumerate(PROTOS):
        d2 = df.copy(); d2["rotation_protocol"] = pr
        if pr == "naive_overwrite":
            d2["old_credential_grace_s"] = 0
        S2, C2 = sequences(d2, p_all)
        pp, _ = gru_predict(mfull, S2, C2)
        for j, c in enumerate(CRASH):
            cf[i, j] = pp[(df.injected_crash_point == c).values].mean()
    pd.DataFrame(cf, index=PROTOS, columns=CRASH).to_csv(os.path.join(RES, "counterfactual_protocol_risk_gru.csv"))

    # ------------------------------------------------------------------ figures
    def save(fig, name):
        fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(RES, name + ".jpg"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Fig 2: learning curves
    L = pd.DataFrame(logs)
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.75))
    for f in range(1, N_FOLDS + 1):
        s = L[L.fold == f]
        ax[0].plot(s.epoch, s.train_loss, color="#c0392b", alpha=0.35 + 0.13 * f, lw=1)
        ax[1].plot(s.epoch, s.val_pr_auc, color="#c0392b", alpha=0.35 + 0.13 * f, lw=1, label=f"fold {f}")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Train loss, Eq. (7)")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Val. PR-AUC")
    ax[1].legend(fontsize=6, frameon=False, ncol=1, loc="lower right")
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(pad=0.3, w_pad=1.2); save(fig, "fig2_learning_curves")

    # Fig 3: main comparison bars
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    mm = ["F1", "PR_AUC", "ROC_AUC", "FailTx_capture_top25"]; lab = ["F1", "PR-AUC", "ROC-AUC", "Failed-tx\ncapture@25%"]
    w = 0.2
    for i, k in enumerate(COL):
        s = T3[T3.method == k].iloc[0]
        ax.bar(np.arange(4) + (i - 1.5) * w, [s[x + "_mean"] for x in mm], w, yerr=[s[x + "_std"] for x in mm],
               color=COL[k], label=k, capsize=1.5, error_kw={"lw": 0.7})
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(lab); ax.set_ylabel("Score (mean ± std, 5 folds)")
    lo = min(T3[[x + "_mean" for x in mm]].values.min() - 0.08, 0.6)
    ax.set_ylim(lo, 1.0); ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=7)
    fig.tight_layout(pad=0.3); save(fig, "fig3_main_comparison")

    # Fig 4: PR curves out-of-fold
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for k in COL:
        pr, rc, _ = precision_recall_curve(y_all, oof[k])
        ax.plot(rc, pr, color=COL[k], lw=1.4 if "GRU" in k else 1.0,
                label=f"{k} (AP={average_precision_score(y_all, oof[k]):.3f})")
    ax.axhline(y_all.mean(), color="k", ls=":", lw=0.8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0.2, 1.01); ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=6.8, loc="lower left", bbox_to_anchor=(0.0, 0.17))
    fig.tight_layout(pad=0.3); save(fig, "fig4_pr_curves")

    # Fig 5: per-crash-point PR-AUC
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    short = ["after\ncreate", "after\nstore", "after\ntarget", "during\nreload", "before\nrevoke"]
    for i, k in enumerate(COL):
        ax.bar(np.arange(5) + (i - 1.5) * w, PER[k].values, w, color=COL[k], label=k)
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(short, fontsize=7)
    ax.set_ylabel("PR-AUC (out-of-fold)"); ax.set_ylim(max(0.0, PER[list(COL)].values.min() - 0.05), min(1.0, PER[list(COL)].values.max() + 0.03)); ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=7)
    fig.tight_layout(pad=0.3); save(fig, "fig5_per_crash_point")

    # Fig 6: observed vs GRU counterfactual heatmap
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.95), sharey=True)
    pl = ["naive", "2PC", "dual-cred.", "vers. lease"]; cl = ["none", "create", "store", "target", "reload", "revoke"]
    for a, M, tt in [(ax[0], pc.values, "Observed rate"), (ax[1], cf, "GRU-CR counterfactual")]:
        im = a.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        a.set_xticks(range(6)); a.set_xticklabels(cl, rotation=60, fontsize=6.5); a.set_title(tt, fontsize=8)
        for i in range(4):
            for j in range(6):
                a.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5.6, color="white" if M[i, j] > 0.55 else "black")
    ax[0].set_yticks(range(4)); ax[0].set_yticklabels(pl, fontsize=7)
    fig.text(0.55, 0.0, "Injected crash point", ha="center", fontsize=8)
    fig.tight_layout(pad=0.3, rect=(0, 0.04, 1, 1)); save(fig, "fig6_protocol_crash_heatmap")

    # Fig 7: ablation + sensitivity
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.8), gridspec_kw={"width_ratios": [1.1, 1]})
    al = ["Full", "no h0\nctx", "λ=0", "order-\nfree"]
    ax[0].bar(range(4), T4.PR_AUC_mean, yerr=T4.PR_AUC_std, color=["#c0392b", "#e59a8f", "#e59a8f", "#e59a8f"], capsize=2)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(al, fontsize=6.8)
    ax[0].set_ylabel("PR-AUC"); vmin = T4.PR_AUC_mean.min(); ax[0].set_ylim(vmin - 0.03, T4.PR_AUC_mean.max() + 0.02)
    ax[0].grid(axis="y", alpha=0.3)
    ax[1].errorbar(hidden_grid, T5.PR_AUC_mean, yerr=T5.PR_AUC_std, color="#c0392b", marker="o", ms=3, capsize=2, lw=1, label="PR-AUC")
    ax[1].errorbar(hidden_grid, T5.F1_mean, yerr=T5.F1_std, color="#555555", marker="s", ms=3, capsize=2, lw=1, label="F1")
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(hidden_grid); ax[1].set_xticklabels(hidden_grid)
    ax[1].set_xlabel("GRU hidden size H"); ax[1].legend(frameon=False, fontsize=6.5); ax[1].grid(alpha=0.3)
    fig.tight_layout(pad=0.3, w_pad=1.6); save(fig, "fig7_ablation_sensitivity")

    shutil.copy(os.path.join(FIG, "fig1_flowchart.png"), os.path.join(RES, "fig1_flowchart.png"))
    from PIL import Image
    Image.open(os.path.join(FIG, "fig1_flowchart.png")).convert("RGB").save(os.path.join(RES, "fig1_flowchart.jpg"), quality=95)
    os.remove(os.path.join(RES, "fig1_flowchart.png"))

    metrics_json = {
        "main": {r["method"]: {k: float(v) for k, v in r.items() if k != "method"} for r in T3.to_dict("records")},
        "ablation": {r["variant"]: {k: float(v) for k, v in r.items() if k != "variant"} for r in T4.to_dict("records")},
        "sensitivity_hidden": {int(r["hidden"]): {k: float(v) for k, v in r.items() if k != "hidden"} for r in T5.to_dict("records")},
        "paired_t_tests": tests,
        "per_crash_point_pr_auc": PER.to_dict("records"),
        "aux_head_inconsistency_type_on_inconsistent_runs": {"macro_F1": float(aux_macro_f1), "accuracy": float(aux_acc_all)},
        "generator_oracle_reference": oracle,
        "oof_pr_auc": {k: float(average_precision_score(y_all, oof[k])) for k in COL},
        "gru_epochs_run_per_fold": [int(L[L.fold == f].epoch.max()) for f in range(1, N_FOLDS + 1)],
        "runtime_s": time.time() - t0,
    }
    json.dump(metrics_json, open(os.path.join(RES, "metrics.json"), "w"), indent=2)
    print(T3.round(4).to_string()); print(T4.round(4).to_string()); print(T5.round(4).to_string())
    print(json.dumps(tests, indent=1)); print(PER.round(3).to_string())
    print("aux", aux_macro_f1, aux_acc_all, "oracle", oracle, "runtime", time.time() - t0)


if __name__ == "__main__":
    main()
