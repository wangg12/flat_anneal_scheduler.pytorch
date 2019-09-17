## Flat and anneal lr scheduler in pytorch

`warmup_method`:
* `linear`
* `constant`

`anneal_method`:

* `cosine`
* `(multi-)step`
* `poly`
* `linear`
* `exp`

## Usage:
See `test_flat_and_anneal()`.

## Convention
* The scheduler should be applied by iteration (or by batch) instead of by epoch.
* `anneal_point` and `steps` are the percentages of the total iterations.
* `init_warmup_lr = warmup_factor * base_lr`
* `target_lr = target_lr_factor * base_lr`
