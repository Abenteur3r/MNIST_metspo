"""Historical v7 numerical routines, retained only as a benchmark reference."""

from collections import deque
import numpy as np

NOBODY = -1
N_MOVE_EXIT = 3
FPS = 25
TEAM_HOME, TEAM_AWAY = 0, 1
V1_D, STATE_HOME, STATE_AWAY = 0, 1, 2
LETTERS = {V1_D: "D", STATE_HOME: "H", STATE_AWAY: "A"}
REENTER_L = 10
V_STILL = 0.5
STILL_SCALE = 0.0
AGE_CAP = 350
CONTEST_W = 10
CONTEST_K = 3.0

PARAMS = dict(
    # D-логика: те же физические константы, что у v1 (не менять без вердикта).
    d_ctrl=6.0, d_weak=15.0, v_restart=2.0, n_move=5, look=10,
    gap_long=np.inf,  # выключено (ablation v1: вредит spellF1)
    # HMM:
    tau_s=12.0,        # априор длительности spell, c (10–20 по sources/01)
    # emissions (log-правдоподобия)
    e_strong_own=1.0, e_strong_opp=-1.0,
    e_weak_own=0.4, e_weak_opp=-0.15,
    e_far_own=0.1, e_far_opp=0.0,
    v_aerial=8.0,      # выше — мяч в воздухе: доказательство ×0.5
    aerial_scale=0.5,
    init_bias=2.0,     # приоритет команде restart'а в первом кадре сегмента
)

PARAMS2B = dict(PARAMS, age_cap=AGE_CAP, contest_w=CONTEST_W, contest_k=CONTEST_K)


def v2_emissions(teams, dists, ball_seen, v, p=None):
    """Матрица (2, n) log-emissions для состояний (H, A)."""
    q = dict(PARAMS)
    if p:
        q.update(p)
    n = len(teams)
    vis = ball_seen
    strong = vis & (dists < q["d_ctrl"])
    weak = vis & (dists >= q["d_ctrl"]) & (dists < q["d_weak"])
    far = vis & (dists >= q["d_weak"])
    scale = np.where(vis & (v > q["v_aerial"]), q["aerial_scale"], 1.0)
    e = np.zeros((2, n), dtype=np.float64)
    for si, tc in ((0, TEAM_HOME), (1, TEAM_AWAY)):
        own = vis & (teams == tc)
        opp = vis & (teams != tc)
        e[si] = ((own & strong) * q["e_strong_own"] + (own & weak) * q["e_weak_own"]
                 + (own & far) * q["e_far_own"]
                 + (opp & strong) * q["e_strong_opp"] + (opp & weak) * q["e_weak_opp"]
                 + (opp & far) * q["e_far_opp"]) * scale
    return e


def v5_emissions(teams, dists, ball_seen, v, p=None):
    """Копия v2_emissions × scale: scale=0 на неподвижном видимом мяче."""
    q = dict(PARAMS)
    if p:
        q.update(p)
    e = v2_emissions(teams, dists, ball_seen, v, q)
    still = ball_seen & (v <= V_STILL)
    if still.any():
        e[:, still] *= STILL_SCALE
    return e


def contest_mask(teams, dists, ball_seen, d_ctrl, w=CONTEST_W):
    """True, если в последних w видимых кадрах strong (dists<d_ctrl) есть у обеих команд."""
    n = len(teams)
    strong = ball_seen & (dists < d_ctrl)
    out = np.zeros(n, dtype=bool)
    dq = deque()
    c0 = c1 = 0
    for f in range(n):
        if ball_seen[f]:
            if len(dq) == w:
                t, s = dq.popleft()
                if s:
                    if t == 0:
                        c0 -= 1
                    else:
                        c1 -= 1
            t = int(teams[f])
            sf = bool(strong[f])
            dq.append((t, sf))
            if sf:
                if t == 0:
                    c0 += 1
                else:
                    c1 += 1
        out[f] = c0 >= 1 and c1 >= 1
    return out


def _viterbi_step(V, e0, e1, log_stay, lt_age, cl, k_max):
    """One forward frame: switch into age 1, grow every other age."""
    sw0 = V[1] + lt_age - cl
    sw1 = V[0] + lt_age - cl
    a0 = int(np.argmax(sw0))
    a1 = int(np.argmax(sw1))
    newV = np.empty_like(V)
    newV[0, 0] = e0 + sw0[a0]  # age 1 только через переход
    newV[1, 0] = e1 + sw1[a1]
    newV[0, 1:] = e0 + V[0, :-1] + log_stay
    newV[1, 1:] = e1 + V[1, :-1] + log_stay
    # старейший бакет (age >= k_max) поглощающий: путь дольше K кадров не теряется
    so0, se0 = V[0, -1] + log_stay, V[0, -2] + log_stay
    so1, se1 = V[1, -1] + log_stay, V[1, -2] + log_stay
    old0 = 1 if so0 >= se0 else 0
    old1 = 1 if so1 >= se1 else 0
    newV[0, -1] = e0 + (so0 if so0 >= se0 else se0)
    newV[1, -1] = e1 + (so1 if so1 >= se1 else se1)
    return newV, a0, a1, old0, old1


def _viterbi_trace(V, bp, bp_old, n, k_max):
    """Backtrack the best final state through the recorded switch ages."""
    path = np.empty(n, dtype=np.int8)
    s = 0 if V[0].max() >= V[1].max() else 1
    a = int(np.argmax(V[s]))
    path[n - 1] = s
    for f in range(n - 1, 0, -1):
        if 0 < a < k_max - 1:
            a -= 1
        elif a == 0:
            a = bp[s, f]
            s = 1 - s
        else:  # старейший бакет
            a = k_max - 1 if bp_old[s, f] == 1 else k_max - 2
        path[f - 1] = s
    return path


def viterbi_age(e, log_stay, log_turn, age_gain, init, k_max=AGE_CAP,
                contest_log=0.0, contested=None):
    """Точный Viterbi для 2 состояний с age-зависимой p перехода.

    e: (2, n) log-emissions; init: (2,). log p_switch(age) =
    log_turn + age_gain*min(age, k_max) - (contest_log, если contested[f]).
    age = число кадров в текущем состоянии (1 на первом кадре).
    age_gain=0 и contest_log=0 -> идентично viterbi2 (самопроверка ниже).
    """
    n = e.shape[1]
    if n == 0:
        return np.empty(0, dtype=np.int8)
    ages = np.arange(1, k_max + 1, dtype=np.float64)
    lt_age = log_turn + age_gain * np.minimum(ages, k_max)
    V = np.full((2, k_max), -np.inf)
    V[0, 0] = init[0] + e[0, 0]
    V[1, 0] = init[1] + e[1, 0]
    bp = np.zeros((2, n), dtype=np.int32)   # age-индекс источника перехода в (s, age 1) на f
    bp_old = np.zeros((2, n), dtype=np.int8)  # 1: старейший бакет пришёл из себя, 0: из K-2
    for f in range(1, n):
        cl = contest_log if (contested is not None and contested[f]) else 0.0
        V, bp[0, f], bp[1, f], bp_old[0, f], bp_old[1, f] = _viterbi_step(
            V, e[0, f], e[1, f], log_stay, lt_age, cl, k_max)
    return _viterbi_trace(V, bp, bp_old, n, k_max)


def _enter_dead(state, reenter):
    """Clear the observation windows and mark the current state dead."""
    state["is_dead"] = True
    state["reenter"] = reenter
    state["vis_idx"] = []
    state["vis_move"] = []
    state["still"] = []


def _still_quiet(state, value):
    """Track the trailing quiet run; True once it reaches the re-entry length."""
    still = state["still"]
    still.append(value <= V_STILL)
    if len(still) > REENTER_L:
        still.pop(0)
    return len(still) == REENTER_L and all(still)


def _trim(vis_idx, vis_move, look):
    """Keep only the newest ``look`` visible frames."""
    while len(vis_idx) > look:
        vis_idx.pop(0)
        vis_move.pop(0)


def _majority(cands):
    """Most frequent team, or None when no frame carried a team."""
    return max(set(cands), key=cands.count) if cands else None


def _restart_team(teams, dists, ball_seen, q, f, value, state):
    """Record one visible frame; return the new restart team once motion exits."""
    vis_idx, vis_move = state["vis_idx"], state["vis_move"]
    vis_idx.append(f)
    vis_move.append(value > (V_STILL if state["reenter"] else q["v_restart"]))
    _trim(vis_idx, vis_move, q["look"])
    # H12b (единственная добавка к v6): min_n для не-reenter D.
    min_n = 5 if state["reenter"] else N_MOVE_EXIT
    if len(vis_move) < min_n or sum(vis_move) < min_n:
        return None
    cands = [int(teams[i]) for i in vis_idx if ball_seen[i] and dists[i] < q["d_ctrl"]]
    state["is_dead"] = False
    state["reenter"] = False
    return _majority(cands)


def _advance(state, teams, dists, ball_seen, v, gapd, q, f, velocity_valid):
    """One frame of the dead/restart state machine; return a new restart team."""
    if gapd[f] and not state["is_dead"]:
        _enter_dead(state, False)
    elif ball_seen[f] and velocity_valid[f] and dists[f] < q["d_ctrl"]:
        if not state["is_dead"] and _still_quiet(state, v[f]):
            _enter_dead(state, True)
        if state["is_dead"]:
            return _restart_team(teams, dists, ball_seen, q, f, v[f], state)
    return None


def v7_dead_timeline(teams, dists, ball_seen, v, dead, gapd, reset_frames, p=None,
                     velocity_valid=None):
    """Копия v6_dead_timeline + H12b: выход из НЕ-reenter D по 3/10."""
    q = dict(PARAMS)
    if p:
        q.update(p)
    n = len(teams)
    if velocity_valid is None:
        velocity_valid = np.isfinite(v)
    dead_out = np.zeros(n, dtype=bool)
    restart_team = np.full(n, NOBODY, dtype=np.int8)
    state = {"is_dead": True, "reenter": False,
             "vis_idx": [], "vis_move": [], "still": []}
    dead_start = dead & np.concatenate(([True], ~dead[:-1])) if n else dead

    for f in range(n):
        if f in reset_frames or dead_start[f]:
            _enter_dead(state, False)
        team = _advance(state, teams, dists, ball_seen, v, gapd, q, f, velocity_valid)
        if team is not None:
            restart_team[f] = team
        dead_out[f] = state["is_dead"]
    return dead_out, restart_team


def _label_run(state, e, run, restart, settings):
    """Label one live run of frames, seeded by its restart team."""
    q, log_stay, log_turn, age_gain, contest_log, contested = settings
    if restart == TEAM_HOME:
        init = (q["init_bias"], 0.0)
    elif restart == TEAM_AWAY:
        init = (0.0, q["init_bias"])
    else:
        init = (0.0, 0.0)
    path = viterbi_age(e[:, run], log_stay, log_turn, age_gain, init,
                       k_max=q["age_cap"], contest_log=contest_log,
                       contested=None if contested is None else contested[run])
    state[run] = np.where(path == 0, STATE_HOME, STATE_AWAY)


def _label_block(state, e, dead_mask, restart_team, first, last, settings):
    """Label the live runs of one graded block; dead frames keep V1_D."""
    seg = np.arange(first, last + 1)
    idx = np.flatnonzero(~dead_mask[seg])
    if len(idx) == 0:
        return
    split = np.flatnonzero(np.diff(idx) > 1) + 1
    for start, stop in zip(np.r_[0, split], np.r_[split, len(idx)]):
        run = seg[idx[start:stop]]
        _label_run(state, e, run, int(restart_team[run[0]]), settings)


def v7_labels(teams, dists, ball_seen, v, dead, gapd, reset_frames,
              graded_blocks_list, n, p=None, use_age=True, use_contest=True,
              velocity_valid=None):
    """Копия v6_labels с v7_dead_timeline; emissions — дословно v5."""
    q = dict(PARAMS2B)
    if p:
        q.update(p)
    dead_mask, restart_team = v7_dead_timeline(teams, dists, ball_seen, v,
                                               dead, gapd, reset_frames, q,
                                               velocity_valid=velocity_valid)
    observed_v = np.where(velocity_valid, v, np.nan) if velocity_valid is not None else v
    e = v5_emissions(teams, dists, ball_seen, observed_v, q)
    tau_frames = max(2, int(round(q["tau_s"] * FPS)))
    log_stay = np.log(1.0 - 1.0 / tau_frames)
    log_turn = np.log(1.0 / tau_frames)
    age_gain = (np.log(2.0) / q["age_cap"]) if use_age else 0.0
    if use_contest:
        contested = contest_mask(teams, dists, ball_seen, q["d_ctrl"],
                                 q["contest_w"])
        contest_log = np.log(q["contest_k"])
    else:
        contested = None
        contest_log = 0.0
    state = np.full(n, V1_D, dtype=np.int8)
    settings = (q, log_stay, log_turn, age_gain, contest_log, contested)
    for first, last in graded_blocks_list:
        _label_block(state, e, dead_mask, restart_team, first, last, settings)
    return state
