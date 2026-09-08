"""Minimal two-agent REINFORCE trainer for MultiRoundHunt's contract."""
import argparse, json, time
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import one_hot, gumbel_softmax


def oh(x, n): return one_hot(x, n).float()


class Agent(nn.Module):
    def __init__(self, private_dim, vocab, action_dim, marker_dim=3, body=None):
        super().__init__()
        self.body = (body if body is not None else
                     nn.Sequential(nn.Linear(private_dim + marker_dim + vocab, 32), nn.Tanh()))
        self.token = nn.Linear(32, vocab)
        self.action = nn.Linear(32, action_dim)
        self.value = nn.Linear(32, 1)

    def forward(self, private, incoming):
        h = self.body(torch.cat((private, incoming), -1))
        return self.token(h), self.action(h), self.value(h).squeeze(-1)


def _fixed_grid_eval(a, b, task, rounds=2, vocab=8, zones=4, type_count=3):
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
                            ta, _, _ = a(pa, last_b)
                            tb, _, _ = b(pb, last_a)
                            ma = ta.argmax(-1)
                            mb = tb.argmax(-1) if task == 'symmetric' else None
                            if task == 'symmetric':
                                last_a, last_b = oh(mb, vocab), oh(ma, vocab)
                            else:
                                last_a = torch.zeros_like(last_a)
                                last_b = oh(ma, vocab)
                            _, aa, _ = a(pa, last_b)
                            _, ab, _ = b(pb, last_a)
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


def run(seed=0, episodes=3000, rounds=2, vocab=8, zones=4, use_messages=True,
        message_temperature=0.7, differentiable_messages=False,
        communication_task='symmetric', curriculum=False, type_count=3,
        coupled=False):
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
            nn.Linear(type_count + zones + rounds + vocab, 32), nn.Tanh())
        a = Agent(type_count + zones, vocab, zones * type_count, rounds,
                  body=shared_body)
        b = Agent(type_count + zones, vocab, zones * type_count, rounds,
                  body=shared_body)
    else:
        a, b = (Agent(type_count + zones, vocab, zones * type_count, rounds),
                Agent(type_count + zones, vocab, zones * type_count, rounds))
    params = []
    seen = set()
    for parameter in list(a.parameters()) + list(b.parameters()):
        if id(parameter) not in seen:
            seen.add(id(parameter)); params.append(parameter)
    opt = torch.optim.Adam(params, lr=.003)
    records=[]; recent=[]; start=time.perf_counter()
    for ep in range(1, episodes+1):
        prey_t, prey_z = torch.randint(type_count,(1,)), torch.randint(zones,(1,))
        trap_t, trap_z = torch.randint(type_count,(1,)), torch.randint(zones,(1,))
        last_a = torch.zeros(1,vocab); last_b = torch.zeros(1,vocab)
        logs=[]; values=[]
        active_task = ('one_way' if curriculum and ep <= episodes // 2
                       else communication_task)
        for r in range(rounds):
            marker = oh(torch.tensor([r]), rounds)
            pa = torch.cat((oh(prey_t,type_count), oh(prey_z,zones), marker), -1)
            pb = torch.cat((oh(trap_t,type_count), oh(trap_z,zones), marker), -1)
            ta, _, _ = a(pa, last_b)
            tb, _, _ = b(pb, last_a)
            if use_messages:
                if differentiable_messages:
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
                if active_task == 'symmetric':
                    last_a, last_b = ((mb, ma) if differentiable_messages
                                      else (oh(mb, vocab), oh(ma, vocab)))
                else:
                    last_a = torch.zeros_like(last_a)
                    last_b = ma if differentiable_messages else oh(ma, vocab)
            else:
                last_a, last_b = torch.zeros_like(last_a), torch.zeros_like(last_b)
                message_logs = []
            # Action is chosen after exchange.  The type component is a
            # deliberate communication bottleneck: A must guess trap_t and B
            # must guess prey_t.
            _, aa, va = a(pa, last_b)
            _, ab, vb = b(pb, last_a)
            daction, baction = Categorical(logits=aa), Categorical(logits=ab)
            act_a, act_b = daction.sample(), baction.sample()
            logs.append(sum(message_logs) + daction.log_prob(act_a) + baction.log_prob(act_b))
            values.append((va+vb)/2)
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
        loss = sum(-log * (ret-val.detach()) + .5*(val-ret).square() for log,val in zip(logs,values)) / rounds
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
            'coupled': coupled,
            'seconds':time.perf_counter()-start,'history':records,
            'fixed_grid_eval': _fixed_grid_eval(a, b, eval_task, rounds, vocab, zones, type_count)}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,default=3000)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); p.add_argument('--output',default='results/multiround-smoke.json')
    p.add_argument('--task',choices=['symmetric','one_way'],default='symmetric')
    p.add_argument('--curriculum',action='store_true')
    p.add_argument('--coupled',action='store_true')
    p.add_argument('--type-count',type=int,default=3); p.add_argument('--zones',type=int,default=4)
    p.add_argument('--rounds',type=int,default=2)
    args=p.parse_args(); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); all=[]
    for s in args.seeds:
        for m in (True,False):
            r=run(s,args.episodes,rounds=args.rounds,zones=args.zones,type_count=args.type_count,use_messages=m,communication_task=args.task,curriculum=args.curriculum,coupled=args.coupled); all.append(r); out.write_text(json.dumps(all,indent=2)); print(json.dumps({'seed':s,'messages':m,'task':args.task,'curriculum':args.curriculum,'coupled':args.coupled,'last':r['history'][-1],'eval':r['fixed_grid_eval']}),flush=True)

if __name__ == '__main__': main()
