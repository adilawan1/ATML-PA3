| dataset   | method                 |   required |   acc@4 |   acc@8 |   acc@32 |   tokens@32 | Δ vs SC @32 (pp) [95% CI]   |
|:----------|:-----------------------|-----------:|--------:|--------:|---------:|------------:|:----------------------------|
| math      | FirstAns               |        1.0 |    64.3 |    64.3 |     64.3 |       669.8 | nan                         |
| math      | Max self-certainty     |        1.0 |    68.8 |    70.1 |     70.6 |     21347.0 | -3.2 [-5.5, -0.8]           |
| math      | Max log-prob           |        1.0 |    67.2 |    67.5 |     64.0 |     21347.0 | nan                         |
| math      | Self-certainty + RWC   |        1.0 |    69.7 |    72.9 |     75.5 |     21347.0 | +1.8 [+0.7, +2.9]           |
| math      | Log-prob + RWC         |        0.0 |    68.5 |    72.2 |     74.9 |     21347.0 | nan                         |
| math      | Self-consistency       |        1.0 |    69.2 |    71.8 |     73.8 |     21347.0 | nan                         |
| math      | Shortest rollout       |        0.0 |    65.2 |    64.6 |     62.4 |     21347.0 | -11.4 [-14.1, -9.0]         |
| math      | Oracle coverage        |      nan   |    79.1 |    84.3 |     91.4 |       nan   | nan                         |
| mathvista | FirstAns               |        1.0 |    53.6 |    53.6 |     53.6 |       277.8 | nan                         |
| mathvista | Max self-certainty     |        1.0 |    55.1 |    54.9 |     54.8 |      8897.8 | -6.9 [-9.9, -3.7]           |
| mathvista | Max log-prob           |        1.0 |    53.3 |    52.0 |     48.2 |      8897.8 | nan                         |
| mathvista | Max vision             |        1.0 |    54.5 |    54.3 |     53.8 |      8897.8 | nan                         |
| mathvista | Max VisConf            |        1.0 |    56.3 |    56.3 |     57.8 |      8897.8 | nan                         |
| mathvista | Max tokenwise fusion   |        0.0 |    55.9 |    55.7 |     56.2 |      8897.8 | nan                         |
| mathvista | Self-certainty + RWC   |        1.0 |    56.9 |    60.1 |     62.5 |      8897.8 | +0.8 [-0.5, +2.2]           |
| mathvista | Log-prob + RWC         |        0.0 |    55.7 |    58.5 |     60.8 |      8897.8 | nan                         |
| mathvista | Vision + RWC           |        1.0 |    56.6 |    60.2 |     63.0 |      8897.8 | +1.3 [-0.2, +3.0]           |
| mathvista | VisConf + RWC          |        1.0 |    57.8 |    60.8 |     63.2 |      8897.8 | +1.6 [+0.2, +3.1]           |
| mathvista | Tokenwise fusion + RWC |        0.0 |    57.6 |    60.9 |     62.7 |      8897.8 | nan                         |
| mathvista | Self-consistency       |        1.0 |    57.8 |    59.9 |     61.7 |      8897.8 | nan                         |
| mathvista | Shortest rollout       |        0.0 |    55.2 |    54.6 |     54.0 |      8897.8 | -7.7 [-10.7, -4.8]          |
| mathvista | Oracle coverage        |      nan   |    75.9 |    82.7 |     89.5 |       nan   | nan                         |