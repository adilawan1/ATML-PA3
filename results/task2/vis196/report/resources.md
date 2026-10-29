| phase                                  |   time_min |   peak_GB |   trainable_params |   ms_per_answer |
|:---------------------------------------|-----------:|----------:|-------------------:|----------------:|
| feature caching (SigLIP, separate)     |       0.61 |      1.36 |               0.00 |          nan    |
| train: caption_alignment               |      18.17 |      1.55 |         1837056.00 |          nan    |
| train: visual_instruction              |      18.37 |      1.53 |         6424576.00 |          nan    |
| train: natural_vqa                     |       9.19 |      1.55 |         6424576.00 |          nan    |
| final eval generation (mean ms/answer) |     nan    |      1.90 |             nan    |           21.91 |