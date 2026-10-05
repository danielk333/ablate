# Paper comparison references

Figures 3 and 4 are extracted without alteration from the authors' version 2
of Dimant & Oppenheim (2017), *Formation of plasma around a small meteoroid:
2. Implications for radar head echo*, doi:10.1002/2017JA023963.

Source: https://open.bu.edu/bitstreams/67a5df26-98ed-4fa5-af3a-a338397ceedd/download
(PDF pages 19 and 20, including the repository cover sheet).

Run `conda run -n base python docs/examples/compare_dimant_oppenheim_paper.py`
to compare these with MetAblate's own density functions. The script requires
`h5py` for its HDF5 data output; neither the model nor its pytest regression
test requires h5py or these images.

The Figure 3 caption normalizes M with q replaced by lambda_T/R (Eq. 17),
equivalent to pi*ne/n_star. The Figure 4 caption specifies relative units
without defining a scale; 8*pi*ne/n_star is an explicit display choice, not
an independently established physical density. The original Figure 4
colorbar is sampled between its printed 10^-3 and 10^4 ticks for matching
colors. The reference images retain attribution to the article's authors.

`tests/test_dimant_oppenheim_paper.py` checks MetAblate's grid morphology and
physical-density scaling against fixed, independently calculated values
of Eqs. 45-48 at all nine Figure 3 angles. It neither downloads the paper
nor uses image pixels as numerical reference data.

## Before and after the fix

`conda run -n base python docs/examples/compare_dimant_before_after.py` loads
the unchanged density module from upstream commit `c31ef4e` through Git and
compares it with the module in the current checkout. It writes PNG plots and
HDF5 data to `dimant_oppenheim_before_after/`, including source hashes and
commit provenance. It requires Git history containing that upstream commit
and h5py. Numerical warnings from the original implementation are recorded;
its invalid-integral handling is preserved without modification.

The bundled `figure3_before_after.png` shows all nine Figure 3 angles with
identical axes before and after. The original forward-axis curve has an
artificial near-field dip; the corrected curve is smooth and matches the
paper comparison. `figure4_before_after.png` uses identical density color
limits for both spatial maps. Both figures include the relative change,
defined as `100 * (after - before) / after`; the map's color limits are
minus to plus 100 percent. These maps use `ne/n_star` directly, without
the optional Figure 4 display multiplier used in the article comparison.
