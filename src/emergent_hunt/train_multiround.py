"""Minimal two-agent REINFORCE trainer for MultiRoundHunt's contract."""
import argparse, json, time
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import one_hot, gumbel_softmax, cross_entropy


def oh(x, n): return one_hot(x, n).float()


def _route_messages(ma, mb, vocab, task, use_messages=True):
    """Return (last_a, last_b), i.e. input for B and A respectively."""
    if not use_messages:
        return torch.zeros(1, vocab), torch.zeros(1, vocab)
    if task == 'symmetric':
        return oh(ma, vocab), oh(mb, vocab)
    return oh(ma, vocab), torch.zeros(1, vocab)


class Agent(nn.Module):
    def __init__(self, private_dim, vocab, action_dim, marker_dim=3, type_count=3,
                 hidden_dim=32, body=None):
        super().__init__()
        self.body = (body if body is not None else
                     nn.Sequential(nn.Linear(private_dim + marker_dim + vocab, hidden_dim), nn.Tanh()))
        self.token = nn.Linear(hidden_dim, vocab)
        self.message_decoder = nn.Linear(vocab, type_count)
        self.zone_action = nn.Linear(hidden_dim + type_count, action_dim // type_count)
        self.type_action = nn.Linear(hidden_dim + type_count, type_count)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, private, incoming):
        h = self.body(torch.cat((private, incoming), -1))
        message_type_logits = self.message_decoder(incoming)
        action_h = torch.cat((h, message_type_logits.softmax(-1)), -1)
        zone_logits = self.zone_action(action_h)
        type_logits = self.type_action(action_h)
        joint_logits = (zone_logits.unsqueeze(-1) + type_logits.unsqueeze(-2)).reshape(
            zone_logits.shape[0], -1)
        return self.token(h), joint_logits, self.value(h).squeeze(-1), message_type_logits


def _fixed_grid_eval(a, b, task, rounds=2, vocab=8, zones=4, type_count=3,
                     use_messages=True):
    """Deterministic argmax evaluation over every latent state."""
    totals = {'zone_score': 0.0, 'type_score': 0.0, 'terminal_success': 0.0}
    count = type_count * zones * type_count * zones
    with torch.no_grad():
        for prey_t in range(type_count):
            for prey_z in range(zones):
                for trap_t in range(type_count):
                    for trap_z in range(zones):
                        last_a = torch.zeros(1, vocab)
                        last_b = torch.zeros(1, vocab)
                        for r in range(rounds):
                            marker = oh(torch.tensor([r]), rounds)
                            pa = torch.cat((oh(torch.tensor([prey_t]), type_count),
                                            oh(torch.tensor([prey_z]), zones), marker), -1)
                            pb = torch.cat((oh(torch.tensor([trap_t]), type_count),
                                            oh(torch.tensor([trap_z]), zones), marker), -1)
                            ta, _, _, _ = a(pa, last_b)
                            tb, _, _, _ = b(pb, last_a)
                            ma = ta.argmax(-1)
                            mb = tb.argmax(-1) if task == 'symmetric' else None
                            last_a, last_b = _route_messages(ma, mb, vocab, task, use_messages)
                            _, aa, _, _ = a(pa, last_b)
                            _, ab, _, _ = b(pb, last_a)
                            act_a, act_b = aa.argmax(-1), ab.argmax(-1)
                        a_zone, a_guess = act_a // type_count, act_a % type_count
                        b_zone, b_guess = act_b // type_count, act_b % type_count
                        totals['zone_score'] += (0.5 * ((a_zone == prey_z).float() +
                                                         (b_zone == trap_z).float())).item()
                        if task == 'one_way':
                            totals['type_score'] += (b_guess == prey_t).float().item()
                            totals['terminal_success'] += ((a_zone == prey_z) &
                                                           (b_zone == trap_z) &
                                                           (b_guess == prey_t)).float().item()
                        else:
                            totals['type_score'] += (0.5 * ((a_guess == trap_t).float() +
                                                            (b_guess == prey_t).float())).item()
                            totals['terminal_success'] += ((a_zone == prey_z) &
                                                           (b_zone == trap_z) &
                                                           (a_guess == trap_t) &
                                                           (b_guess == prey_t) &
                                                           (prey_t != trap_t)).float().item()
    return {k: v / count for k, v in totals.items()}


def _protocol_diagnostics(a, b, task, rounds=2, vocab=8, zones=4, type_count=3,
                          use_messages=True):
    """Separate production purity, oracle comprehension and do(token) effect."""
    rows = []
    with torch.no_grad():
        for prey_t in range(type_count):
            for prey_z in range(zones):
                for trap_t in range(type_count):
                    for trap_z in range(zones):
                        last_a = torch.zeros(1, vocab); last_b = torch.zeros(1, vocab)
                        for r in range(rounds):
                            marker = oh(torch.tensor([r]), rounds)
                            pa = torch.cat((oh(torch.tensor([prey_t]), type_count), oh(torch.tensor([prey_z]), zones), marker), -1)
                            pb = torch.cat((oh(torch.tensor([trap_t]), type_count), oh(torch.tensor([trap_z]), zones), marker), -1)
                            ta, _, _, _ = a(pa, last_b); tb, _, _, _ = b(pb, last_a)
                            ma = ta.argmax(-1); mb = tb.argmax(-1) if task == 'symmetric' else None
                            last_a, last_b = _route_messages(ma, mb, vocab, task, use_messages)
                            _, aa, _, _ = a(pa, last_b); _, ab, _, _ = b(pb, last_a)
                        rows.append((prey_t, trap_t, ma.item(), mb.item() if mb is not None else -1))

    def purity(label_index, token_index):
        score = 0.0
        for label in range(type_count):
            tokens = [r[token_index] for r in rows if r[label_index] == label]
            counts = [tokens.count(k) for k in range(vocab)]
            score += max(counts) / max(1, len(tokens))
        return score / type_count

    def dominant_tokens(label_index, token_index):
        result = []
        for label in range(type_count):
            tokens = [r[token_index] for r in rows if r[label_index] == label]
            result.append(max(set(tokens), key=tokens.count))
        return result

    prey_to_a = dominant_tokens(0, 2)
    trap_to_b = []
    if task == 'symmetric':
        trap_to_b = dominant_tokens(1, 3)

    oracle_hits = 0.0; oracle_total = 0; sensitivity = 0.0
    with torch.no_grad():
        for prey_t in range(type_count):
            for prey_z in range(zones):
                for trap_t in range(type_count):
                    for trap_z in range(zones):
                        marker = oh(torch.tensor([rounds - 1]), rounds)
                        pa = torch.cat((oh(torch.tensor([prey_t]), type_count), oh(torch.tensor([prey_z]), zones), marker), -1)
                        pb = torch.cat((oh(torch.tensor([trap_t]), type_count), oh(torch.tensor([trap_z]), zones), marker), -1)
                        if task == 'one_way':
                            _, ab, _, _ = b(pb, oh(torch.tensor([prey_to_a[prey_t]]), vocab))
                            oracle_hits += (ab.argmax(-1) % type_count == prey_t).float().item()
                            guesses = {int(b(pb, oh(torch.tensor([k]), vocab))[1].argmax(-1).item() % type_count) for k in range(vocab)}
                        else:
                            _, aa, _, _ = a(pa, oh(torch.tensor([trap_to_b[trap_t]]), vocab))
                            _, ab, _, _ = b(pb, oh(torch.tensor([prey_to_a[prey_t]]), vocab))
                            oracle_hits += 0.5 * ((aa.argmax(-1) % type_count == trap_t).float().item() + (ab.argmax(-1) % type_count == prey_t).float().item())
                            guesses = {int(b(pb, oh(torch.tensor([k]), vocab))[1].argmax(-1).item() % type_count) for k in range(vocab)}
                        oracle_total += 1
                        sensitivity += len(guesses) / type_count
    return {'production_purity_sender_a': purity(0, 2),
            'production_injective_sender_a': len(set(prey_to_a)) / type_count,
            'production_purity_sender_b': (purity(1, 3) if task == 'symmetric' else None),
            'production_injective_sender_b': (len(set(trap_to_b)) / type_count if task == 'symmetric' else None),
            'oracle_receiver_type_accuracy': oracle_hits / oracle_total,
            'receiver_token_sensitivity': sensitivity / oracle_total}


def run(seed=0, episodes=3000, rounds=2, vocab=8, zones=4, use_messages=True,
        message_temperature=0.7, differentiable_messages=False,
        communication_task='symmetric', curriculum=False, type_count=3,
        coupled=False, receiver_aux=0.0, sender_aux=0.0,
        receiver_bootstrap_episodes=0, hidden_dim=32):
    if communication_task not in ('symmetric', 'one_way'):
        raise ValueError('communication_task must be symmetric or one_way')
    torch.manual_seed(seed); torch.set_num_threads(1)
    # private = type (3) + zone (4); incoming = last token (8) + round marker (3)
    # An action is (own zone, guess of the partner's hidden type).  The
    # previous zone-only head made communication causally irrelevant: each
    # agent already observed its own zone, while type incompatibility was not
    # controllable by any action.
    if type_count < 2 or zones < 2 or rounds < 1:
        raise ValueError('type_count, zones and rounds must be >= 2, 2, and 1')
    if coupled:
        shared_body = nn.Sequential(
            nn.Linear(type_count + zones + rounds + vocab, hidden_dim), nn.Tanh())
        a = Agent(type_count + zones, vocab, zones * type_count, rounds,
                  type_count=type_count, hidden_dim=hidden_dim, body=shared_body)
        b = Agent(type_count + zones, vocab, zones * type_count, rounds,
                  type_count=type_count, hidden_dim=hidden_dim, body=shared_body)
    else:
        a, b = (Agent(type_count + zones, vocab, zones * type_count, rounds,
                      type_count=type_count, hidden_dim=hidden_dim),
                Agent(type_count + zones, vocab, zones * type_count, rounds,
                      type_count=type_count, hidden_dim=hidden_dim))
    params = []
    seen = set()
    for parameter in list(a.parameters()) + list(b.parameters()):
        if id(parameter) not in seen:
            seen.add(id(parameter)); params.append(parameter)
    opt = torch.optim.Adam(params, lr=.003)
    if receiver_aux < 0 or sender_aux < 0:
        raise ValueError('auxiliary coefficients must be nonnegative')
    if receiver_bootstrap_episodes < 0:
        raise ValueError('receiver_bootstrap_episodes must be nonnegative')
    state_rng = torch.Generator().manual_seed(seed + 1000003)
    records=[]; recent=[]; start=time.perf_counter()
    for ep in range(1, episodes+1):
        prey_t = torch.randint(type_count, (1,), generator=state_rng)
        prey_z = torch.randint(zones, (1,), generator=state_rng)
        trap_t = torch.randint(type_count, (1,), generator=state_rng)
        trap_z = torch.randint(zones, (1,), generator=state_rng)
        last_a = torch.zeros(1,vocab); last_b = torch.zeros(1,vocab)
        logs=[]; values=[]; aux_losses=[]; sender_losses=[]
        active_task = ('one_way' if curriculum and ep <= episodes // 2
                       else communication_task)
        for r in range(rounds):
            marker = oh(torch.tensor([r]), rounds)
            pa = torch.cat((oh(prey_t,type_count), oh(prey_z,zones), marker), -1)
            pb = torch.cat((oh(trap_t,type_count), oh(trap_z,zones), marker), -1)
            ta, _, _, _ = a(pa, last_b)
            tb, _, _, _ = b(pb, last_a)
            if use_messages:
                teacher_forced = ep <= receiver_bootstrap_episodes
                if teacher_forced:
                    # A declared curriculum aid: the receiver first sees a
                    # stable reference code. The hidden type is never added to
                    # an observation and teacher forcing is disabled later.
                    ma = prey_t.clone()
                    mb = trap_t.clone() if active_task == 'symmetric' else None
                    message_logs = []
                elif differentiable_messages:
                    # Optional straight-through control. The default retains
                    # explicit message-policy REINFORCE below.
                    ma = gumbel_softmax(ta, tau=message_temperature, hard=True)
                    mb = (gumbel_softmax(tb, tau=message_temperature, hard=True)
                          if active_task == 'symmetric' else None)
                    message_logs = []
                else:
                    da = Categorical(logits=ta)
                    ma = da.sample()
                    mb = (Categorical(logits=tb).sample()
                          if communication_task == 'symmetric' else None)
                    message_logs = [da.log_prob(ma)]
                    if active_task == 'symmetric':
                        message_logs.append(Categorical(logits=tb).log_prob(mb))
                last_a, last_b = _route_messages(ma, mb, vocab, active_task, True)
            else:
                last_a, last_b = _route_messages(None, None, vocab, active_task, False)
                message_logs = []
            # Action is chosen after exchange.  The type component is a
            # deliberate communication bottleneck: A must guess trap_t and B
            # must guess prey_t.
            _, aa, va, a_decode = a(pa, last_b)
            _, ab, vb, b_decode = b(pb, last_a)
            daction, baction = Categorical(logits=aa), Categorical(logits=ab)
            act_a, act_b = daction.sample(), baction.sample()
            logs.append(sum(message_logs) + daction.log_prob(act_a) + baction.log_prob(act_b))
            values.append((va+vb)/2)
            # Privileged target is used only as an auxiliary training signal;
            # neither agent receives the partner type as an observation.
            b_type_logits = b_decode
            sender_losses.append(cross_entropy(ta, prey_t))
            if active_task == 'symmetric':
                sender_losses.append(cross_entropy(tb, trap_t))
            if active_task == 'one_way':
                aux_losses.append(cross_entropy(b_type_logits, prey_t))
            else:
                a_type_logits = a_decode
                aux_losses.append(0.5 * (cross_entropy(a_type_logits, trap_t) +
                                         cross_entropy(b_type_logits, prey_t)))
        a_zone, a_guess_trap = act_a // type_count, act_a % type_count
        b_zone, b_guess_prey = act_b // type_count, act_b % type_count
        zone_score = 0.5 * ((a_zone == prey_z).float() + (b_zone == trap_z).float())
        if active_task == 'one_way':
            type_score = (b_guess_prey == prey_t).float()
            terminal = ((a_zone == prey_z) & (b_zone == trap_z) &
                        (b_guess_prey == prey_t)).float()
        else:
            type_score = 0.5 * ((a_guess_trap == trap_t).float() +
                                (b_guess_prey == prey_t).float())
            terminal = ((a_zone == prey_z) & (b_zone == trap_z) &
                        (a_guess_trap == trap_t) & (b_guess_prey == prey_t) &
                        (prey_t != trap_t)).float()
        # Dense components make the communication-dependent type prediction
        # learnable under REINFORCE; terminal remains a separately weighted
        # success signal rather than being silently conflated with shaping.
        reward = 0.15 * zone_score + 0.40 * type_score + 0.45 * terminal
        ret = reward.detach()
        policy_loss = sum(-log * (ret-val.detach()) + .5*(val-ret).square()
                          for log,val in zip(logs,values)) / rounds
        effective_aux = receiver_aux if use_messages else 0.0
        effective_sender_aux = sender_aux if use_messages else 0.0
        loss = (policy_loss + effective_aux * sum(aux_losses) / rounds +
                effective_sender_aux * sum(sender_losses) / rounds)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 5); opt.step()
        recent.append(reward.item())
        if len(recent) > 100: recent.pop(0)
        if ep == 1 or ep % 300 == 0 or ep == episodes:
            records.append({'episode':ep,'reward':reward.item(),
                            'reward_mean_100':sum(recent)/len(recent),
                            'zone_score':zone_score.item(),
                            'type_score':type_score.item(),
                            'terminal_success':terminal.item(),
                            'loss':loss.item()})
    eval_task = communication_task
    return {'seed':seed,'episodes':episodes,'rounds':rounds,'use_messages':use_messages,
            'communication_task': communication_task, 'curriculum': curriculum,
            'coupled': coupled, 'receiver_aux': receiver_aux, 'sender_aux': sender_aux,
            'receiver_bootstrap_episodes': receiver_bootstrap_episodes,
            'hidden_dim': hidden_dim,
            'seconds':time.perf_counter()-start,'history':records,
            'fixed_grid_eval': _fixed_grid_eval(a, b, eval_task, rounds, vocab, zones, type_count, use_messages),
            'protocol_diagnostics': _protocol_diagnostics(a, b, eval_task, rounds, vocab, zones, type_count, use_messages)}


def run_staged_one_way(seed=0, sender_episodes=1000, receiver_episodes=2000,
                       joint_episodes=3000, vocab=8, zones=4, type_count=3,
                       coupled=False, hidden_dim=32):
    """Canonical sender -> frozen receiver -> joint one-way curriculum."""
    torch.manual_seed(seed); torch.set_num_threads(1)
    if coupled:
        body = nn.Sequential(nn.Linear(type_count + zones + 2 + vocab, hidden_dim), nn.Tanh())
        a = Agent(type_count + zones, vocab, zones * type_count, 2,
                  type_count=type_count, hidden_dim=hidden_dim, body=body)
        b = Agent(type_count + zones, vocab, zones * type_count, 2,
                  type_count=type_count, hidden_dim=hidden_dim, body=body)
    else:
        a = Agent(type_count + zones, vocab, zones * type_count, 2,
                  type_count=type_count, hidden_dim=hidden_dim)
        b = Agent(type_count + zones, vocab, zones * type_count, 2,
                  type_count=type_count, hidden_dim=hidden_dim)

    def make_params(*modules):
        out = []; seen = set()
        for module in modules:
            for p in module.parameters():
                if id(p) not in seen: seen.add(id(p)); out.append(p)
        return out

    # Phase 1: learn a canonical token (token index == private prey type).
    opt_a = torch.optim.Adam(make_params(a), lr=.003)
    for _ in range(sender_episodes):
        prey_t = torch.randint(type_count, (1,)); prey_z = torch.randint(zones, (1,))
        pa = torch.cat((oh(prey_t, type_count), oh(prey_z, zones), oh(torch.tensor([0]), 2)), -1)
        token_logits, _, _, _ = a(pa, torch.zeros(1, vocab))
        loss = cross_entropy(token_logits, prey_t)
        opt_a.zero_grad(); loss.backward(); opt_a.step()

    # Phase 2: freeze sender and teacher-force its canonical code into B.
    for p in a.parameters(): p.requires_grad_(False)
    opt_b = torch.optim.Adam(make_params(b), lr=.003)
    for _ in range(receiver_episodes):
        prey_t = torch.randint(type_count, (1,)); trap_t = torch.randint(type_count, (1,))
        trap_z = torch.randint(zones, (1,))
        pb = torch.cat((oh(trap_t, type_count), oh(trap_z, zones), oh(torch.tensor([0]), 2)), -1)
        _, action_logits, _, decode_logits = b(pb, oh(prey_t, vocab))
        target_action = trap_z * type_count + prey_t
        loss = cross_entropy(decode_logits, prey_t) + cross_entropy(action_logits, target_action)
        opt_b.zero_grad(); loss.backward(); opt_b.step()

    # Phase 3: restore joint learning with actual sampled sender tokens.
    for p in a.parameters(): p.requires_grad_(True)
    params = make_params(a, b); opt = torch.optim.Adam(params, lr=.001)
    for _ in range(joint_episodes):
        prey_t = torch.randint(type_count, (1,)); prey_z = torch.randint(zones, (1,))
        trap_t = torch.randint(type_count, (1,)); trap_z = torch.randint(zones, (1,))
        pa = torch.cat((oh(prey_t, type_count), oh(prey_z, zones), oh(torch.tensor([0]), 2)), -1)
        pb = torch.cat((oh(trap_t, type_count), oh(trap_z, zones), oh(torch.tensor([0]), 2)), -1)
        ta, _, _, _ = a(pa, torch.zeros(1, vocab)); ma = Categorical(logits=ta).sample()
        _, action_logits, _, decode_logits = b(pb, oh(ma, vocab))
        action = Categorical(logits=action_logits).sample()
        target_action = trap_z * type_count + prey_t
        reward = (action == target_action).float()
        loss = (-Categorical(logits=ta).log_prob(ma) * reward.detach() -
                Categorical(logits=action_logits).log_prob(action) * reward.detach() +
                0.1 * cross_entropy(decode_logits, prey_t))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 5); opt.step()
    return {'seed': seed, 'sender_episodes': sender_episodes,
            'receiver_episodes': receiver_episodes, 'joint_episodes': joint_episodes,
            'fixed_grid_eval': _fixed_grid_eval(a, b, 'one_way', 2, vocab, zones, type_count),
            'protocol_diagnostics': _protocol_diagnostics(a, b, 'one_way', 2, vocab, zones, type_count)}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,default=3000)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); p.add_argument('--output',default='results/multiround-smoke.json')
    p.add_argument('--task',choices=['symmetric','one_way'],default='symmetric')
    p.add_argument('--curriculum',action='store_true')
    p.add_argument('--coupled',action='store_true')
    p.add_argument('--receiver-aux',type=float,default=0.0)
    p.add_argument('--sender-aux',type=float,default=0.0)
    p.add_argument('--receiver-bootstrap',type=int,default=0)
    p.add_argument('--type-count',type=int,default=3); p.add_argument('--zones',type=int,default=4)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--hidden-dim',type=int,default=32)
    args=p.parse_args(); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); all=[]
    for s in args.seeds:
        for m in (True,False):
            r=run(s,args.episodes,rounds=args.rounds,zones=args.zones,type_count=args.type_count,use_messages=m,communication_task=args.task,curriculum=args.curriculum,coupled=args.coupled,receiver_aux=args.receiver_aux,sender_aux=args.sender_aux,receiver_bootstrap_episodes=args.receiver_bootstrap,hidden_dim=args.hidden_dim); all.append(r); out.write_text(json.dumps(all,indent=2)); print(json.dumps({'seed':s,'messages':m,'task':args.task,'curriculum':args.curriculum,'coupled':args.coupled,'hidden_dim':args.hidden_dim,'last':r['history'][-1],'eval':r['fixed_grid_eval'],'protocol':r['protocol_diagnostics']}),flush=True)

if __name__ == '__main__': main()
