"""Minimal two-agent REINFORCE trainer for MultiRoundHunt's contract."""
import argparse, json, time
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import one_hot


def oh(x, n): return one_hot(x, n).float()


class Agent(nn.Module):
    def __init__(self, private_dim, vocab, zones):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(private_dim + 3 + vocab, 32), nn.Tanh())
        self.token = nn.Linear(32, vocab)
        self.action = nn.Linear(32, zones)
        self.value = nn.Linear(32, 1)

    def forward(self, private, incoming):
        h = self.body(torch.cat((private, incoming), -1))
        return self.token(h), self.action(h), self.value(h).squeeze(-1)


def run(seed=0, episodes=3000, rounds=2, vocab=8, zones=4, use_messages=True):
    torch.manual_seed(seed); torch.set_num_threads(1)
    # private = type (3) + zone (4); incoming = last token (8) + round marker (3)
    a, b = Agent(7, vocab, zones), Agent(7, vocab, zones)
    opt = torch.optim.Adam(list(a.parameters()) + list(b.parameters()), lr=.003)
    records=[]; recent=[]; start=time.perf_counter()
    for ep in range(1, episodes+1):
        prey_t, prey_z = torch.randint(3,(1,)), torch.randint(zones,(1,))
        trap_t, trap_z = torch.randint(3,(1,)), torch.randint(zones,(1,))
        last_a = torch.zeros(1,vocab); last_b = torch.zeros(1,vocab)
        logs=[]; values=[]
        for r in range(rounds):
            marker = oh(torch.tensor([r]), 3)
            pa = torch.cat((oh(prey_t,3), oh(prey_z,zones), marker), -1)
            pb = torch.cat((oh(trap_t,3), oh(trap_z,zones), marker), -1)
            ta, _, _ = a(pa, last_b)
            tb, _, _ = b(pb, last_a)
            da, db = Categorical(logits=ta), Categorical(logits=tb)
            ma, mb = da.sample(), db.sample()
            if use_messages:
                last_a, last_b = oh(mb,vocab), oh(ma,vocab)
            else:
                last_a, last_b = torch.zeros_like(last_a), torch.zeros_like(last_b)
            # action is chosen after exchange; each agent knows its own zone.
            _, aa, va = a(pa, last_b)
            _, ab, vb = b(pb, last_a)
            daction, baction = Categorical(logits=aa), Categorical(logits=ab)
            act_a, act_b = daction.sample(), baction.sample()
            logs.append(da.log_prob(ma)+db.log_prob(mb)+daction.log_prob(act_a)+baction.log_prob(act_b))
            values.append((va+vb)/2)
        reward = ((act_a == prey_z) & (act_b == trap_z) & (prey_t != trap_t)).float()
        ret = reward.detach()
        loss = sum(-log * (ret-val.detach()) + .5*(val-ret).square() for log,val in zip(logs,values)) / rounds
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(a.parameters())+list(b.parameters()),5); opt.step()
        recent.append(reward.item())
        if len(recent) > 100: recent.pop(0)
        if ep == 1 or ep % 300 == 0 or ep == episodes:
            records.append({'episode':ep,'reward':reward.item(),'reward_mean_100':sum(recent)/len(recent),'loss':loss.item()})
    return {'seed':seed,'episodes':episodes,'rounds':rounds,'use_messages':use_messages,
            'seconds':time.perf_counter()-start,'history':records}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,default=3000)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); p.add_argument('--output',default='results/multiround-smoke.json')
    args=p.parse_args(); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); all=[]
    for s in args.seeds:
        for m in (True,False):
            r=run(s,args.episodes,use_messages=m); all.append(r); out.write_text(json.dumps(all,indent=2)); print(json.dumps({'seed':s,'messages':m,'last':r['history'][-1]}),flush=True)

if __name__ == '__main__': main()
