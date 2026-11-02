"""Cost of an API request and break-even against an own cluster.

Chapter 16 of "Secrets of DeepSeek AI".

All rates below are ILLUSTRATIVE PARAMETERS in "price units" (PU) per
million tokens, not real prices. API prices change; plug in the current
values from the provider's pricing page before drawing conclusions.

Run:
    python cost_calculator.py
"""
from dataclasses import dataclass


@dataclass
class Rates:
    input_miss: float   # PU per 1M input tokens not found in the context cache
    input_hit: float    # PU per 1M input tokens served from the cache
    output: float       # PU per 1M output tokens (for reasoners: includes the CoT)


def request_cost(r: Rates, t_in, t_out, hit_ratio=0.0):
    """Cost of one request: split input into cache hits and misses."""
    hit = t_in * hit_ratio
    miss = t_in - hit
    return (miss * r.input_miss + hit * r.input_hit + t_out * r.output) / 1e6


def own_cluster_cost_per_mtok(cluster_cost_per_hour, throughput_tok_s, utilization):
    """Cost per 1M generated tokens on own hardware.

    cluster_cost_per_hour – amortized hardware + power + staff, PU/hour
    throughput_tok_s      – aggregate output tokens/s at full load
    utilization           – fraction of the hour the cluster does useful work
    """
    tokens_per_hour = throughput_tok_s * 3600 * utilization
    return cluster_cost_per_hour / tokens_per_hour * 1e6


def break_even_utilization(cluster_cost_per_hour, throughput_tok_s, api_rate):
    """Formula (16.4): utilization at which own tokens cost the API rate."""
    return cluster_cost_per_hour * 1e6 / (3600 * throughput_tok_s * api_rate)


if __name__ == "__main__":
    rates = Rates(input_miss=1.0, input_hit=0.1, output=4.0)   # illustrative
    print("Illustrative rates (PU / 1M tokens):", rates, "\n")

    chat = dict(t_in=3000, t_out=600)
    for h in (0.0, 0.5, 0.9):
        c = request_cost(rates, **chat, hit_ratio=h)
        print(f"chat request, cache hit {h:>4.0%}: {c*1000:7.3f} PU per 1000 requests")

    # a reasoner answers briefly but thinks at length: the CoT is billed as output
    short_answer = request_cost(rates, t_in=800, t_out=300)
    with_thinking = request_cost(rates, t_in=800, t_out=300 + 4000)
    print(f"\nreasoner: answer only {short_answer*1e3:.3f}, with 4000-token CoT "
          f"{with_thinking*1e3:.3f} PU per 1000 requests (x{with_thinking/short_answer:.1f})")

    # monthly workload
    req_per_day = 200_000
    month = request_cost(rates, **chat, hit_ratio=0.5) * req_per_day * 30
    print(f"\nmonthly API bill for {req_per_day} chat requests/day: {month:,.0f} PU")

    # own cluster: the same model, illustrative numbers
    for util in (0.15, 0.4, 0.8):
        c = own_cluster_cost_per_mtok(cluster_cost_per_hour=60.0,
                                      throughput_tok_s=6000, utilization=util)
        print(f"own cluster at {util:>4.0%} utilization: {c:6.2f} PU per 1M "
              f"output tokens (API output rate: {rates.output:.2f})")
    u = break_even_utilization(60.0, 6000, rates.output)
    print(f"break-even utilization u* = {u:.2f}")
