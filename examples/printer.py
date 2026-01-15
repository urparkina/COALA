from math import floor

def beaty(data, digits=2, colw=7):
    def truncate(x, digits=2):
        factor = 10 ** digits
        return floor(x * factor) / factor

    keys = [
        ("arc_challenge", "acc_norm,none", "ARC-C"),
        ("arc_easy",      "acc_norm,none", "ARC-E"),
        ("boolq",         "acc,none",      "BoolQ"),
        ("hellaswag",     "acc_norm,none", "Hella"),
        ("openbookqa",    "acc_norm,none", "OBQA"),
        ("piqa",          "acc_norm,none", "PIQA"),
        ("winogrande",    "acc,none",      "Wino"),
    ]

    header = [f"{abbr:>{colw}}" for _, _, abbr in keys]
    header.append(f"{'Mean':>{colw}}")
    print(" | ".join(header) + " |")

    acc_values = []
    values = []
    for name, metric, _ in keys:
        val = data[name][metric]
        acc_values.append(val)
        values.append(truncate(val * 100, digits))

    mean_acc = sum(acc_values) / len(acc_values)
    values.append(truncate(mean_acc * 100, digits))

    row = [f"{v:>{colw}.{digits}f}" for v in values]
    print(" | ".join(row) + " |")
