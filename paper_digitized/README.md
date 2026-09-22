# Digitized curves from the paper's Figs. 11, 12, 13

Extracted from the PDF rendered at 500 dpi (`fig1{1,2,3}_hi.png`) by `digitize.py`:
ggplot panel located by its grey fill; gridlines = narrow white runs inside it; pixel-to-data
map from the equal gridline spacing anchored on the first labelled tick (Fig. 11) or on the
ADR reference line, which is exactly 0 by definition (Figs. 12/13); series identified by
nearest legend colour; each series read as the median row of its pixels in a +-4 px column
at every query x.

Validation against numbers printed in the text: ADR 83.1/88.8/94.5/97.3/99.0 % at LM 6/7/8/9/11
(paper: >=80/85/90/95/99 at those LMs); ANN ToA 32.7 % (text 32.7), energy 43.3 % (text 43.5);
SVR 30.2/41.2 (29.9/40.6); RF 27.6/38.6 (27.5/38.7). Read-out precision about +-2 points.
Known bad cells: Fig. 11 FRIIS at LM 3-8 (marker overlapped by other series); Fig. 13 ADR at
PDR 90 (annotation arrow); values slightly above 100 are marker-height noise.
