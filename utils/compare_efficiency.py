import argparse
import json


def metric(summary, name):
    value = summary["metrics"][name]["mean"]
    if value is None:
        raise ValueError("Metric {!r} is unavailable".format(name))
    return float(value)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and static-route reports")
    parser.add_argument("baseline", help="baseline efficiency_summary.json")
    parser.add_argument("optimized", help="optimized efficiency_summary.json")
    args = parser.parse_args()

    with open(args.baseline, "r") as handle:
        baseline = json.load(handle)
    with open(args.optimized, "r") as handle:
        optimized = json.load(handle)

    train_base = metric(baseline, "cl_train_seconds")
    train_opt = metric(optimized, "cl_train_seconds")
    route_base = metric(baseline, "routed_training_seconds")
    route_opt = metric(optimized, "routed_training_seconds")
    infer_base = metric(baseline, "inference_seconds_per_sample")
    infer_opt = metric(optimized, "inference_seconds_per_sample")

    s_cl_train = ratio(train_base, train_opt)
    s_route = ratio(route_base, route_opt)
    p = ratio(route_base, train_base)
    result = {
        "S_CL_train": s_cl_train,
        "S_route": s_route,
        "p": p,
        "S_overall_Amdahl": 1.0 / ((1.0 - p) + p / s_route),
        "S_infer": ratio(infer_base, infer_opt),
    }
    for flops_name in ("routed_train_flops_per_sample", "inference_flops_per_sample"):
        try:
            base_flops = metric(baseline, flops_name)
            opt_flops = metric(optimized, flops_name)
            result[flops_name + "_reduction"] = 1.0 - ratio(opt_flops, base_flops)
        except (KeyError, ValueError):
            result[flops_name + "_reduction"] = None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
