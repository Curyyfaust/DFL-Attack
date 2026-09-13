"""Concept-selective attacks for the wind-forecast-to-OPF pipeline.

The attack direction combines a learned economic-sensitivity direction with
the gradient of one named operational concept. Gradients of the two non-target
concepts are projected out in forecast space, making concept selectivity an
explicit, testable part of the attack rather than a post-hoc visualization.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

import run_experiment as base
from run_cbm_experiment import batch_concepts, concept_features


CONCEPT_NAMES = ["Net-load", "Network", "Ramping"]


def network_features(pred_cf, timestamps):
    """Differentiable polynomial features for the network-stress surrogate."""
    pred_cf = np.asarray(pred_cf)
    ts = pd.to_datetime(np.asarray(timestamps).ravel())
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    wind = pred_cf.ravel()
    load = np.array([base.load_multiplier(t) for t in ts])
    sin_h = np.sin(2 * np.pi * hour / 24)
    cos_h = np.cos(2 * np.pi * hour / 24)
    return np.column_stack([
        wind, load, wind * load, wind ** 2, load ** 2,
        sin_h, cos_h, wind * sin_h, wind * cos_h,
    ])


def surrogate_concepts(pred_cf, timestamps, network_model, smooth=1e-4):
    """Three differentiable concepts; shape: samples x horizon x concepts."""
    pred_cf = np.asarray(pred_cf)
    ts = pd.to_datetime(np.asarray(timestamps).ravel())
    load_mult = np.array([base.load_multiplier(t) for t in ts]).reshape(pred_cf.shape)
    net_load = (
        base.PD_MW.sum() * load_mult - pred_cf * base.WIND_CAP
    ) / base.GEN_CAP.sum()
    network = network_model.predict(network_features(pred_cf, timestamps)).reshape(pred_cf.shape)
    temporal_change = np.gradient(net_load, axis=1)
    ramping = np.sqrt(temporal_change ** 2 + smooth ** 2)
    return np.stack([net_load, network, ramping], axis=-1)


def concept_summary_gradients(pred_cf, timestamps, network_model, step=2e-4):
    """Finite-difference gradients of mean concepts w.r.t. forecast horizons."""
    n, horizon = pred_cf.shape
    gradients = np.empty((n, 3, horizon))
    for h in range(horizon):
        plus = pred_cf.copy()
        minus = pred_cf.copy()
        plus[:, h] = np.clip(plus[:, h] + step, 0, 1)
        minus[:, h] = np.clip(minus[:, h] - step, 0, 1)
        cp = surrogate_concepts(plus, timestamps, network_model).mean(axis=1)
        cm = surrogate_concepts(minus, timestamps, network_model).mean(axis=1)
        denominator = (plus[:, h] - minus[:, h])[:, None] + 1e-12
        gradients[:, :, h] = (cp - cm) / denominator
    return gradients


def unit_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def selective_direction(concept_grads, economic_grad, target, economic_weight=0.1):
    """Project target/economic direction onto the off-target null space."""
    n, _, horizon = concept_grads.shape
    target_grad = unit_rows(concept_grads[:, target, :])
    economic = unit_rows(economic_grad)
    raw = target_grad + economic_weight * economic
    directions = np.empty((n, horizon))
    off_idx = [j for j in range(3) if j != target]
    for i in range(n):
        off = concept_grads[i, off_idx, :]
        off = off / (np.linalg.norm(off, axis=1, keepdims=True) + 1e-12)
        gram = off @ off.T + 1e-5 * np.eye(len(off_idx))
        projected = raw[i] - off.T @ np.linalg.solve(gram, off @ raw[i])
        # Preserve the requested positive target-concept direction.
        if projected @ concept_grads[i, target] < 0:
            projected = -projected
        directions[i] = projected / (np.mean(np.abs(projected)) + 1e-12)
    return directions


def concept_attack(model, x, timestamps, network_model, economic_grad,
                   target, eps, steps=10, alpha=None):
    """Projected attack in input-gradient space.

    Projection is applied after propagation through the forecasting-model
    Jacobian. This avoids losing concept orthogonality when forecast-space
    directions are mapped back to the historical input.
    """
    alpha = eps / 4 if alpha is None else alpha
    xa = x.copy()
    for _ in range(steps):
        pred, cache = model.forward(xa, cache=True)
        concept_grads = concept_summary_gradients(pred, timestamps, network_model)
        input_concept_grads = []
        for k in range(3):
            _, gx_k = model.backward(cache, concept_grads[:, k, :])
            input_concept_grads.append(gx_k)
        input_concept_grads = np.stack(input_concept_grads, axis=1)
        _, economic_grad_x = model.backward(cache, economic_grad)
        direction_x = selective_direction(
            input_concept_grads, economic_grad_x, target
        )
        xa = xa + alpha * direction_x
        xa = np.minimum(np.maximum(xa, x - eps), x + eps)
        xa = np.clip(xa, 0, 1)
    return xa


def main():
    data = pd.read_csv("wind_dataset_repo/XJ1907.csv")
    data["date"] = pd.to_datetime(data["date"])
    cf = np.clip(data["Target"].to_numpy(float) / 200.0, 0, 1)
    x, y, timestamps = base.make_windows(cf, data["date"].to_numpy(), 48, 16)
    n = len(x)
    ntr, nv = int(0.70 * n), int(0.15 * n)
    xtr, ytr = x[:ntr], y[:ntr]
    xv, yv, tv = x[ntr:ntr + nv], y[ntr:ntr + nv], timestamps[ntr:ntr + nv]
    xte, yte, tte = x[ntr + nv:], y[ntr + nv:], timestamps[ntr + nv:]

    model = base.MLP(48, 64, 16, seed=base.SEED)
    model.train(xtr, ytr, xv, yv)

    # Offline labels: every second validation window.
    cal_idx = np.arange(0, len(xv), 2)
    pred_cal = model.forward(xv[cal_idx])
    _, sensitivity_cal, concepts_cal = batch_concepts(pred_cal, tv[cal_idx])

    concept_encoder = RandomForestRegressor(
        n_estimators=160, max_depth=9, min_samples_leaf=3,
        random_state=base.SEED, n_jobs=-1,
    )
    concept_encoder.fit(
        concept_features(pred_cal, tv[cal_idx]), concepts_cal.reshape(-1, 3)
    )
    sensitivity_readout = DecisionTreeRegressor(
        max_depth=4, min_samples_leaf=20, random_state=base.SEED
    )
    sensitivity_readout.fit(concepts_cal.reshape(-1, 3), sensitivity_cal.ravel())

    # A differentiable surrogate is used only for concept gradients. Its target
    # remains the OPF-derived network-stress concept.
    network_model = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1e-4),
    )
    network_model.fit(
        network_features(pred_cal, tv[cal_idx]), concepts_cal[..., 1].ravel()
    )
    network_r2 = r2_score(
        concepts_cal[..., 1].ravel(),
        network_model.predict(network_features(pred_cal, tv[cal_idx])),
    )

    xev, yev, tev = xte[-192:], yte[-192:], tte[-192:]
    pred_clean = model.forward(xev)
    clean_cost, sensitivity_true, concepts_clean = batch_concepts(pred_clean, tev)
    concepts_pred = concept_encoder.predict(
        concept_features(pred_clean, tev)
    ).reshape(concepts_clean.shape)
    sensitivity_pred = sensitivity_readout.predict(
        concepts_pred.reshape(-1, 3)
    ).reshape(sensitivity_true.shape)

    eps = 0.02
    attacks = {
        "FGSM": base.fgsm(model, xev, yev, eps),
        "PGD": base.pgd(model, xev, yev, eps, steps=10),
        "SPGA": base.spga(model, xev, sensitivity_true, eps, steps=10),
        "CB-SPGA": base.spga(model, xev, sensitivity_pred, eps, steps=10),
    }
    for k, name in enumerate(["Net-CA", "Grid-CA", "Ramp-CA"]):
        attacks[name] = concept_attack(
            model, xev, tev, network_model, sensitivity_pred,
            target=k, eps=eps, steps=10,
        )

    concept_scale = np.std(concepts_clean, axis=(0, 1)) + 1e-12
    mse0 = float(np.mean((pred_clean - yev) ** 2))
    cost0 = float(np.mean(clean_cost))
    rows = [{
        "Attack": "No attack", "MSE": mse0, "dMSE_%": 0.0,
        "Cost_$": cost0, "dCost_%": 0.0,
        "dNet_std": 0.0, "dGrid_std": 0.0, "dRamp_std": 0.0,
        "Selectivity": np.nan,
    }]
    stored = {"No attack": pred_clean}
    target_map = {"Net-CA": 0, "Grid-CA": 1, "Ramp-CA": 2}
    for name, xa in attacks.items():
        pred = model.forward(xa)
        costs, _, concepts = batch_concepts(pred, tev)
        delta_concepts = np.mean(
            (concepts - concepts_clean) / concept_scale, axis=(0, 1)
        )
        selectivity = np.nan
        if name in target_map:
            k = target_map[name]
            selectivity = abs(delta_concepts[k]) / (
                np.sum(np.abs(delta_concepts)) + 1e-12
            )
        mse = float(np.mean((pred - yev) ** 2))
        cost = float(np.mean(costs))
        rows.append({
            "Attack": name,
            "MSE": mse,
            "dMSE_%": (mse / mse0 - 1) * 100,
            "Cost_$": cost,
            "dCost_%": (cost / cost0 - 1) * 100,
            "dNet_std": delta_concepts[0],
            "dGrid_std": delta_concepts[1],
            "dRamp_std": delta_concepts[2],
            "Selectivity": selectivity,
        })
        stored[name] = pred

    results = pd.DataFrame(rows)
    results.to_csv("concept_attack_results.csv", index=False)
    metrics = {
        "epsilon": eps,
        "network_surrogate_validation_R2": float(network_r2),
        "concept_standard_deviations": dict(zip(CONCEPT_NAMES, concept_scale.tolist())),
        "concept_attack_definition": (
            "economic-plus-target direction projected onto the local "
            "off-target concept null space"
        ),
    }
    Path("concept_attack_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        "concept_attack_arrays.npz",
        timestamps=tev.astype("datetime64[ns]").astype(str),
        y_true=yev,
        concepts_clean=concepts_clean,
        sensitivity_true=sensitivity_true,
        sensitivity_pred=sensitivity_pred,
        **{f"pred_{name.replace('-', '_')}": value for name, value in stored.items()},
    )
    print(results.round(6).to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
