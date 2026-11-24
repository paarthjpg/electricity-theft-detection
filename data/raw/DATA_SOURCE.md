# Dataset Source and Attribution

**Dataset:** SGCC Electricity Theft Detection Dataset
**Original source:** https://github.com/henryRDlab/ElectricityTheftDetection
(split archive: `data.zip` + `data.z01` + `data.z02`, downloaded 2026-11-24)

**Citation:**
Zheng, Z., Yang, Y., Niu, X., Dai, H.-N., & Zhou, Y. (2018). Wide and Deep
Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart
Grids. *IEEE Transactions on Industrial Informatics*, 14(4), 1606-1615.

**Verified file:** `sgcc_data.csv`
**SHA-256:** `99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7`

**Verified shape:** 42,372 consumers x 1,034 daily consumption columns
(2014-01-01 to 2016-10-31), plus `CONS_NO` (customer ID) and `FLAG` (label)
columns.

**Verified label distribution:** 38,757 normal consumers (`FLAG=0`), 3,615
flagged theft consumers (`FLAG=1`) — matches the characteristics reported in
the original paper and subsequent literature.

This is real data from the cited public source, not synthetic data. A small
synthetic fixture used only for fast software smoke tests lives under
`tests/fixtures/` and must never be presented as SGCC results.
