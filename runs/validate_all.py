# Correctness gate: reference logits via decode path and via padded prefill; then compile-cache probe.
exec(open('/content/runs/first_logits.py').read().replace("exec(open('/content/runs/load.py').read(),globals())","exec(open('/content/runs/load.py').read(),globals())"),globals())
exec(open('/content/runs/prefill_check.py').read(),globals())
