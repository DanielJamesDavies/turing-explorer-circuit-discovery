Use `32x32` for the next `counterfactual_gradient` discovery run.

Recommended config:

```yaml
discovery:
  probe_batch_size: 4
  counterfactual_gradient:
    max_neg_sequences: 32
    neg_batch_size: 32
```

Reason: the H100 pilot on 128 seeds found `32x32` had similar speed to the
completed `16x16` run, stayed safely within memory, and slightly improved
acceptance rate. A `64x32` pilot was viable but slower and did not improve
acceptance on the same sample.
