"""REINFORCE trainer for TrapPrepHunt: learned driver/preparer, no oracle.

Naming, because it inverts train.py's: here the *driver* is the informed sender
(it sees every prey) and the *preparer* is the blind receiver (it sees only its
own position, the trap under it, and the clock). There is also a second,
reverse channel -- the preparer's one-token readiness reply, which the driver
hears before it commits its own action in the same timestep -- so the loss
carries four log-probability terms per step, not two.

Both channels are genuinely discrete: tokens are sampled from a Categorical and
one-hot encoded onto the wire, so no gradient flows sender->receiver. Credit
assignment is the shared discounted return, as in train_line.py; the message at
step t is delivered before the action at step t, so message and action log-probs
share the same advantage with no off-by-one (unlike train_line, where messages
arrive a step late).

Both agents are memoryless MLPs, matching train.py's architecture. That is a
real modelling choice, not an oversight: the reference protocol repeats the
instruction every step, so a memoryless preparer is sufficient for it. If a
learned protocol needs to say something once and have it remembered, this
architecture cannot express it -- see the notes in the results file.

Controls, at matched episode budget:
  communication  -- both channels intact.
  no_message     -- the driver->preparer wire is zeroed (the blind condition
                    whose exact optimum is trap_prep.blind_reference).
  no_readiness   -- the preparer->driver wire is zeroed: the driver must time
                    the arrival without hearing that the trap is ready.

No claim of emergent grammar is made from anything here. Catch rate is reported
against trap_prep.blind_reference and the split-matched held-out corpus, and
the learned pair is handed to trap_probe.selectivity for a token-level readout.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import one_hot

from .trap_prep import (ACTIVE, CAUGHT, DRIVE_BASE, PREPARE_BASE, TrapPrepHunt,
                        blind_preparation_bound, blind_reference, mechanism_for,
                        tasks)
from .train import mlp

MODES = ('communication', 'no_message', 'no_readiness')


def _one_hot(index, size):
    return one_hot(torch.tensor([index]).clamp(0, size - 1), size).float()


class TrapEncoder:
    """Fixed one-hot encodings of the two private views and of the wire.

    Nothing here is learned and nothing crosses the privacy boundary: the
    driver encoding never mentions the preparer and vice versa.
    """

    def __init__(self, env):
        self.n, self.targets = env.n, env.targets
        self.mechanisms, self.horizon, self.window = env.mechanisms, env.horizon, env.window
        self.vocabulary = env.vocabulary
        self.driver_slots = env.driver_message_length
        self.preparer_slots = env.preparer_message_length
        self.driver_dim = (self.horizon + 1) + self.targets * (
            2 * self.n + (self.n + 2) + (self.window + 1) + 3)
        self.preparer_dim = (self.horizon + 1) + self.n + (self.mechanisms + 1) + 2
        self.driver_wire = self.preparer_slots * (self.vocabulary + 1)
        self.preparer_wire = self.driver_slots * (self.vocabulary + 1)
        self.driver_actions = DRIVE_BASE + self.targets
        self.preparer_actions = PREPARE_BASE + self.mechanisms

    def driver(self, observation):
        parts = [_one_hot(observation['steps_left'], self.horizon + 1)]
        for j in range(self.targets):
            parts += [_one_hot(observation['prey_type'][j], self.n),
                      _one_hot(observation['zone'][j], self.n),
                      _one_hot(observation['distance'][j], self.n + 2),
                      _one_hot(observation['window_left'][j], self.window + 1),
                      _one_hot(observation['status'][j], 3)]
        return torch.cat(parts, -1)

    def preparer(self, observation):
        return torch.cat([_one_hot(observation['steps_left'], self.horizon + 1),
                          _one_hot(observation['self_pos'], self.n),
                          _one_hot(observation['trap_method'] + 1, self.mechanisms + 1),
                          _one_hot(observation['trap_charged'], 2)], -1)

    def wire(self, tokens, slots):
        """Token t occupies index t+1; an erased token (-1) index 0; a missing
        slot is all zeros, so message length remains a channel."""
        parts = []
        for i in range(slots):
            if i < len(tokens):
                parts.append(_one_hot(tokens[i] + 1, self.vocabulary + 1))
            else:
                parts.append(torch.zeros(1, self.vocabulary + 1))
        return torch.cat(parts, -1)


class TrapDriver(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.message_head = mlp(encoder.driver_dim,
                                encoder.driver_slots * encoder.vocabulary)
        self.action_head = mlp(encoder.driver_dim + encoder.driver_wire,
                               encoder.driver_actions)
        self.critic = mlp(encoder.driver_dim, 1)

    def message_logits(self, x):
        return self.message_head(x).view(-1, self.encoder.driver_slots,
                                         self.encoder.vocabulary)

    def action_logits(self, x, wire):
        return self.action_head(torch.cat((x, wire), -1))


class TrapPreparer(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        width = encoder.preparer_dim + encoder.preparer_wire
        self.message_head = mlp(width, encoder.preparer_slots * encoder.vocabulary)
        self.action_head = mlp(width, encoder.preparer_actions)
        self.critic = mlp(width, 1)

    def message_logits(self, x):
        return self.message_head(x).view(-1, self.encoder.preparer_slots,
                                         self.encoder.vocabulary)


def _zero_if(wire, muted):
    return torch.zeros_like(wire) if muted else wire


def rollout(env, driver, preparer, encoder, task=None, mode='communication',
            greedy=False):
    """One episode. Returns rewards and the log-probs/entropies of all four
    decision streams, so the caller can build any credit-assignment scheme."""
    mute_driver = mode == 'no_message'
    mute_preparer = mode == 'no_readiness'
    driver_obs, preparer_obs = env.reset(task)
    rewards, logps, entropies, values, trace = [], [], [], [], []
    done = False
    while not done:
        dx = encoder.driver(driver_obs)
        message_dist = Categorical(logits=driver.message_logits(dx))
        message = (message_dist.logits.argmax(-1) if greedy
                   else message_dist.sample())
        preparer_obs, heard = env.driver_send(tuple(int(t) for t in message[0]))
        px = torch.cat((encoder.preparer(preparer_obs),
                        _zero_if(encoder.wire(heard, encoder.driver_slots), mute_driver)), -1)
        reply_dist = Categorical(logits=preparer.message_logits(px))
        reply = reply_dist.logits.argmax(-1) if greedy else reply_dist.sample()
        driver_obs, back = env.preparer_send(tuple(int(t) for t in reply[0]))
        dx2 = encoder.driver(driver_obs)
        action_logits = driver.action_logits(
            dx2, _zero_if(encoder.wire(back, encoder.preparer_slots), mute_preparer))
        driver_dist = Categorical(logits=action_logits)
        driver_action = driver_dist.logits.argmax(-1) if greedy else driver_dist.sample()
        env.driver_act(int(driver_action))
        preparer_dist = Categorical(logits=preparer.action_head(px))
        preparer_action = (preparer_dist.logits.argmax(-1) if greedy
                           else preparer_dist.sample())
        reward, done, info = env.preparer_act(int(preparer_action))

        logp = driver_dist.log_prob(driver_action) + preparer_dist.log_prob(preparer_action)
        entropy = driver_dist.entropy() + preparer_dist.entropy()
        if not mute_driver:
            logp = logp + message_dist.log_prob(message).sum(-1)
            entropy = entropy + message_dist.entropy().sum(-1)
        if not mute_preparer:
            logp = logp + reply_dist.log_prob(reply).sum(-1)
            entropy = entropy + reply_dist.entropy().sum(-1)
        rewards.append(reward)
        logps.append(logp.squeeze(0))
        entropies.append(entropy.squeeze(0))
        values.append((driver.critic(dx).squeeze() + preparer.critic(px).squeeze()) / 2)
        trace.append(info)
    return {'rewards': rewards, 'logps': logps, 'entropies': entropies,
            'values': values, 'info': trace[-1], 'trace': trace}


@torch.no_grad()
def evaluate(driver, preparer, encoder, env, corpus, mode='communication'):
    """Deterministic (argmax) evaluation over a whole split, plus the mute
    intervention on the same policies."""
    result = {}
    for condition in ('intact', 'mute'):
        caught = prepared = prey = reward = success = 0
        first_guess = in_position = 0
        for task in corpus:
            run_mode = mode if condition == 'intact' else 'no_message'
            out = rollout(env, driver, preparer, encoder, task=task,
                          mode=run_mode, greedy=True)
            info = out['info']
            caught += info['caught']
            prey += len(task.targets)
            reward += sum(out['rewards'])
            success += int(info['success'])
            prepared += sum(1 for t in task.targets
                            if env.prepared_methods[t.zone] == mechanism_for(
                                t.prey_type, env.mechanisms))
            # The one-shot preparation payment is the message-sensitive
            # statistic: correct_prep_rate below only reads the trap's FINAL
            # mechanism, which a blind preparer can also set by trying every
            # mechanism in turn. first_guess_rate counts the prey whose trap was
            # right on the single attempt that could pay, so a blind preparer
            # is pinned at 1/mechanisms.
            first_guess += info['totals']['prepare'] / env.partial if env.partial else 0
            in_position += sum(1 for step in out['trace']
                               for name, _ in step['events'] if name == 'in_position')
        result[condition] = {'catch_rate': caught / prey,
                             'first_guess_rate': first_guess / prey,
                             'correct_prep_rate': prepared / prey,
                             'in_position_rate': in_position / prey,
                             'mean_reward': reward / len(corpus),
                             'all_caught_rate': success / len(corpus)}
    return result


def run(seed=0, mode='communication', episodes=20000, batch=16, n=3, targets=1,
        by='pair', horizon=8, window=1, patience=4, approach=0.0, start_pos=0,
        stop_when_resolved=False, gamma=.97, lr=3e-3, entropy_coef=.02,
        report_every=2000, checkpoint=None, on_report=None):
    """Defaults matter here and are not arbitrary.

    patience=4 (with n=3) is the regime in which the reference protocol still
    solves every task while a message-blind sweep cannot reach a second trap, so
    trap_prep.blind_reference is 1/9, 1/6, 1/3 rather than the 1/3, 1/2, 1.0 it
    would be with no deadline. Without it the by='pair' test split is solvable
    blind and a high test catch rate would mean nothing.

    stop_when_resolved=False keeps the scene running after the prey is gone.
    Early termination is a trap for a learner: an untrained driver drives at
    once, the prey escapes on step two, and the preparer never gets enough
    steps to discover that preparing pays at all.

    horizon=8 and start_pos=0 are kept as the historical defaults so the
    published sweep stays reproducible, but the diagnosed regime for new runs is
    horizon=4, start_pos=1, for two independent reasons:

    * horizon > patience + window - 1 leaves steps on which nothing can be won.
      They still emit four log-probability terms each, and per-episode advantage
      normalization gives them a systematically negative advantage, so they are
      not merely wasted -- they push probability mass away from whatever was
      emitted there. Cutting them also removes a freebie: the partial reward
      does not check the prey's status, so at horizon=8 a preparer could stroll
      the whole line arming traps long after the prey had fled, which is exactly
      what the first sweep learned to do. It also tightens the blind ceiling on
      first_guess_rate from 1/2 to 1/3 on the train split.
    * start_pos=1 (the middle of the line) leaves every bound untouched -- a
      blind two-trap sweep still costs 5 steps from anywhere -- while cutting
      the reference protocol's worst-case travel from 2 steps to 1. At
      start_pos=0 the oracle's slack against the deadline is exactly zero on the
      far zone, so learners had to hit a one-step firing window with no margin
      at all. At start_pos=1 the minimum slack is 1.
    """
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    env = TrapPrepHunt(n=n, targets=targets, split='train', by=by, seed=seed,
                       horizon=horizon, window=window, patience=patience,
                       approach=approach, start_pos=start_pos,
                       stop_when_resolved=stop_when_resolved)
    encoder = TrapEncoder(env)
    driver, preparer = TrapDriver(encoder), TrapPreparer(encoder)
    parameters = list(driver.parameters()) + list(preparer.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    train_corpus = tasks(n, targets, 'train', by)
    test_corpus = tasks(n, targets, 'test', by)
    history = []
    start = time.perf_counter()
    initial = evaluate(driver, preparer, encoder, env, train_corpus, mode)
    losses = []
    for episode in range(1, episodes + 1):
        out = rollout(env, driver, preparer, encoder, mode=mode)
        returns, running = [], 0.0
        for reward in reversed(out['rewards']):
            running = reward + gamma * running
            returns.append(running)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        values = torch.stack(out['values'])
        advantage = (returns - values).detach()
        if len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        logps = torch.stack(out['logps'])
        entropy = torch.stack(out['entropies']).mean()
        actor = -(advantage * logps).mean()
        critic = (values - returns).square().mean()
        losses.append(actor + .5 * critic - entropy_coef * entropy)
        if episode % batch == 0:
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            assert torch.isfinite(loss)
            nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
            losses = []
        if episode == 1 or episode % report_every == 0 or episode == episodes:
            row = {'episode': episode,
                   'seconds': time.perf_counter() - start,
                   'episode_reward': sum(out['rewards']),
                   'entropy': float(entropy.detach()),
                   'train': evaluate(driver, preparer, encoder, env, train_corpus, mode),
                   'test': evaluate(driver, preparer, encoder, env, test_corpus, mode)}
            history.append(row)
            if on_report:
                on_report(row, driver, preparer)
    if checkpoint:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'driver': driver.state_dict(), 'preparer': preparer.state_dict(),
                    'config': {'seed': seed, 'mode': mode, 'n': n, 'targets': targets,
                               'by': by, 'horizon': horizon, 'window': window,
                               'patience': patience, 'approach': approach,
                               'start_pos': start_pos}},
                   checkpoint)
    return {'seed': seed, 'mode': mode, 'episodes': episodes, 'batch': batch,
            'n': n, 'targets': targets, 'by': by, 'horizon': horizon,
            'window': window, 'patience': patience, 'approach': approach,
            'start_pos': start_pos, 'stop_when_resolved': stop_when_resolved,
            'gamma': gamma, 'lr': lr,
            'entropy_coef': entropy_coef, 'torch': torch.__version__,
            'seconds': time.perf_counter() - start,
            # The bound must be the one for the configuration actually run: it
            # depends on the horizon, the prey's patience AND the firing window,
            # since a wider window postpones the last useful activation for a
            # blind sweeper exactly as much as it does for the learners.
            'blind_bound': {split: blind_reference(
                n=n, split=split, by=by, horizon=horizon, start_pos=start_pos,
                patience=patience, window=window)['optimal_blind_success']
                for split in ('train', 'test')} if targets == 1 else None,
            # first_guess_rate has its own, different ceiling: the partial
            # reward ignores the prey's position, so a blind preparer is bounded
            # by how many zones it can reach and guess at within the horizon.
            'blind_preparation_bound': {split: blind_preparation_bound(
                n=n, split=split, by=by, horizon=horizon,
                start_pos=start_pos)['upper_bound_first_guess_rate']
                for split in ('train', 'test')} if targets == 1 else None,
            'initial': initial, 'history': history,
            'agents': (driver, preparer, encoder, env)}


def probe_factory(driver, preparer, encoder, mode='communication'):
    """Adapt trained modules to trap_probe's (driver, preparer) policy protocol.

    Decoding is greedy, which the probe requires: it compares two runs that
    differ by one token, so any sampling noise would be read as a causal effect.
    """
    class LearnedDriver:
        def reset(self, observation):
            self._back = ()

        @torch.no_grad()
        def message(self, observation):
            logits = driver.message_logits(encoder.driver(observation))
            return tuple(int(t) for t in logits.argmax(-1)[0])

        @torch.no_grad()
        def act(self, observation, incoming):
            wire = _zero_if(encoder.wire(incoming, encoder.preparer_slots),
                            mode == 'no_readiness')
            return int(driver.action_logits(encoder.driver(observation), wire).argmax(-1))

    class LearnedPreparer:
        def reset(self, observation):
            self._x = None

        def _features(self, observation, incoming):
            return torch.cat((encoder.preparer(observation),
                              _zero_if(encoder.wire(incoming, encoder.driver_slots),
                                       mode == 'no_message')), -1)

        @torch.no_grad()
        def message(self, observation, incoming):
            self._x = self._features(observation, incoming)
            return tuple(int(t) for t in preparer.message_logits(self._x).argmax(-1)[0])

        @torch.no_grad()
        def act(self, observation):
            return int(preparer.action_head(self._x).argmax(-1))

    def factory(n, mechanisms):
        return LearnedDriver(), LearnedPreparer()

    return factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--modes', nargs='+', default=list(MODES), choices=MODES)
    parser.add_argument('--targets', type=int, default=1)
    parser.add_argument('--by', choices=['triple', 'pair'], default='pair')
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--start-pos', dest='start_pos', type=int, default=0)
    parser.add_argument('--approach', type=float, default=0.0,
                        help='shaping reward for a prey standing on a correctly '
                             'armed trap, deducted from the catch reward')
    parser.add_argument('--probe', action='store_true',
                        help='run trap_probe.selectivity on the learned pair')
    parser.add_argument('--output', default='results/trap-smoke.json')
    args = parser.parse_args()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        for mode in args.modes:
            checkpoint = target.with_name(f'{target.stem}-{seed}-{mode}.pt')
            result = run(seed=seed, mode=mode, episodes=args.episodes,
                         targets=args.targets, by=args.by, horizon=args.horizon,
                         patience=args.patience, window=args.window,
                         approach=args.approach, start_pos=args.start_pos,
                         checkpoint=checkpoint)
            driver, preparer, encoder, env = result.pop('agents')
            if args.probe and mode == 'communication':
                from .trap_probe import selectivity
                result['selectivity'] = {
                    split: selectivity(policy=probe_factory(driver, preparer, encoder),
                                       n=env.n, targets=env.targets, by=args.by,
                                       split=split, max_tasks=24,
                                       env_kwargs={'horizon': args.horizon,
                                                   'patience': args.patience,
                                                   'window': args.window,
                                                   'approach': args.approach,
                                                   'start_pos': args.start_pos,
                                                   'stop_when_resolved': False})
                    for split in ('train', 'test')}
            results.append(result)
            target.write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(json.dumps({'seed': seed, 'mode': mode,
                              'seconds': round(result['seconds'], 1),
                              'train': result['history'][-1]['train']['intact'],
                              'test': result['history'][-1]['test']['intact'],
                              'blind_bound': result['blind_bound'],
                              'blind_preparation_bound':
                                  result['blind_preparation_bound']}), flush=True)


if __name__ == '__main__':
    main()
