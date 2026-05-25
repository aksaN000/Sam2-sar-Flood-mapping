# Speaker notes for Part 1 (Aksan)

Target total time: **5 to 6 minutes**. Pace: conversational, ~150 words/minute. Each slide note below is what to *say*, not what to read off the screen.

---

## Slide 1 — Title (~30 sec)

> Good [morning / afternoon]. I'm Aksan Gony Alif, and along with my co-author Mahbubur Rahman, we present *Parameter-Efficient Adaptation of SAM 2 for Sentinel-1 SAR Flood Inundation Mapping in the Indo-Gangetic Region*.
>
> The motivation is direct. Bangladesh floods every monsoon. Optical satellites can't see through monsoon clouds, exactly when flood maps matter most. Radar can. But radar imagery looks nothing like the natural photographs that modern vision models like SAM 2 were trained on. So our question is: can we adapt a billion-mask foundation model to noisy single-channel radar, cheaply, and have it actually generalize to a flood event it has never seen?

---

## Slide 2 — Why this paper exists (~45 sec)

> The four bullets on screen lay out the problem.
>
> Bangladesh floods every monsoon. Optical satellites are unreliable precisely when flood mapping is most urgent, because monsoon clouds block the optical sensors. Sentinel-1 SAR penetrates clouds, is global, and is free.
>
> But SAR is fundamentally different from RGB photography. It's a noisy single-channel logarithmic-intensity signal where bright versus dark means rough versus smooth surface, not red versus blue light. Foundation models like SAM 2 don't transfer to it naturally.
>
> Our research question follows. Can we adapt SAM 2 to radar cheaply, and have it generalize to unseen flood events? The answer, mostly yes, with one important caveat about the cross-polarization channel that my co-author will get to in part 2.

---

## Slide 3 — Three gaps in prior work (~50 sec)

> We frame the paper around three concrete gaps in the literature.
>
> *Pause briefly before each gap.*
>
> First, nobody has compared parameter-efficient fine-tuning methods on SAM-family backbones for SAR floods. The literature has scattered single-method papers — one uses LoRA, another uses AdaptFormer — but no head-to-head comparison. A practitioner can't tell which adapter to pick.
>
> Second, prior SAR-flood uncertainty work reports low Expected Calibration Error and stops there. Aggregate calibration is fine, but in deployment operators want per-pixel uncertainty — *which pixels should I trust*. That's a different question, and nobody tests it directly.
>
> Third, there is no public, reproducible SAR flood test set that is genuinely out-of-distribution by both *geography and time*. Sen1Floods11 holds out one country but all its events are from 2017 to 2019. You can't test generalization to a fresh 2022 event without building your own dataset.
>
> This paper closes all three.

---

## Slide 4 — Methodology: the sweep (~50 sec)

> Here is our full evaluation grid. Five backbones, four PEFT methods, three random seeds per cell.
>
> The principal pair is SAM ViT-Base at 94 million parameters and SAM 2 Hiera-Base-Plus at 81 million parameters. These two get the full cross-product: LoRA, DoRA, Convolutional LoRA, and AdaptFormer.
>
> Then we stretch up to three larger backbones — ViT-Large, ViT-Huge at 636 million, and SAM 2 Hiera-Large — under Convolutional LoRA only, to test whether the small-backbone findings transport at scale.
>
> Total cost across the entire sweep, including the U-Net baseline and the zero-shot baselines, is about three and a half hours on a dual-GPU cloud instance for under ten US dollars. The full sweep is genuinely reproducible at that price.
>
> Uncertainty is added via Monte Carlo dropout with 20 stochastic forward passes per chip.

---

## Slide 5 — Polarimetric pseudo-RGB input (~50 sec)

> Now the SAR-specific bit. Sentinel-1 gives us two polarization channels. *VV* — vertical-transmit, vertical-receive, the co-polarization. And *VH*, the cross-polarized channel.
>
> But SAM 2 was pretrained on three RGB channels. So we have to synthesize a third channel from the two we have. This is what we call a "pseudo-RGB" composition.
>
> We compare three of them. The *ratio* composition puts the log ratio of VV over VH in the third channel. The *diff* composition uses dB-domain subtraction. The *single-polarization* control just replicates VV across all three channels and throws VH away.
>
> A small mathematical note: the ratio and diff compositions are mathematically equivalent by a dB-domain identity. They produce numerically identical inputs to the model. So the meaningful contrast — and this becomes critical in part 2 of the talk — is between *dual-polarization* (ratio or diff) and *single-polarization* (VV alone).

---

## Slide 6 — Four evaluation splits (~45 sec)

> Our evaluation surface has four splits.
>
> Three come from Sen1Floods11. The in-distribution test set with 90 chips across ten countries. The Bolivia held-out split with 15 chips, which is the only truly held-out country in the standard benchmark. And a 12-chip Pakistan subset from the 2010 event — this one is regionally informative but still in-distribution, because Pakistan also appears in the training set.
>
> The fourth split — *highlighted on screen* — is what we built in this work. The Pakistan-2022 set. 39 chips covering the August-September 2022 Indo-Gangetic monsoon flood. This is the same meteorological system that produces Bangladesh's annual flooding, and it is the only strictly out-of-distribution test surface in this evaluation, by both *geography and time*.

---

## Slide 7 — Pakistan-2022 dataset construction (~50 sec)

> The Pakistan-2022 dataset is built from two complementary public sources.
>
> Labels come from a TU Wien Sentinel-1-derived flood-extent product released alongside the Roth et al. NHESS 2023 paper, under a CC-BY license.
>
> Imagery comes from Microsoft Planetary Computer's `sentinel-1-rtc` catalogue — these are Radiometric Terrain Corrected Sentinel-1 GRD products, fully georeferenced, with anonymous SAS-token access. No account needed.
>
> Our pipeline, for each TU Wien mask date: find the matching Sentinel-1 scene, read at native 10 metre resolution, reproject the mask upward to 10 metres, tile into 512 by 512 chips. Result: 39 chips at native 10 metre resolution, fully Sen1Floods11-compatible layout.
>
> One acquisition note we document honestly in the paper: the first chip set was at 20 metres and mismatched the Sen1Floods11 training distribution. We rebuilt at native 10 metres. That's the dataset released in this paper.

---

## Slide 8 — Experimental setup (~30 sec)

> Quick training protocol. Each cell is 10 epochs on the 252 hand-labelled Sen1Floods11 training chips. AdamW optimizer, two learning rates — 10 to the minus four for adapter parameters and 10 to the minus five for any unfrozen pretrained weights. Cosine schedule with 200-step linear warmup. Loss is Dice plus binary cross-entropy, equal weight. Mixed precision throughout. Effective batch size 4.
>
> Important: we do *not* apply training-time data augmentation. With only 252 chips, we want every training signal to come from the actual labels, not from synthetic transformations. Test-time augmentation is used only at inference, only for the ensemble condition.
>
> Three random seeds per cell: 42, 123, 20025.

---

## Slide 9 — Handoff (~20 sec)

> *Pause briefly. Look up at camera.*
>
> Setup complete. The next half of this talk is what we actually found across these four evaluation splits — plus one diagnostic finding we did not expect, which turned out to be the most operationally useful result in the paper.
>
> *Pause. Land the line.*
>
> The cross-polarization channel — the one that helps in-distribution — turns out to be the *operative* out-of-distribution failure mode.
>
> I'll hand it over to my co-author Mahbubur.

---

## Delivery checklist

- [ ] **Slide 1**: state your name and your co-author's name slowly. Look up at camera, not at the slide.
- [ ] **Slide 3**: pause briefly between "First", "Second", and "Third". The three-beat rhythm is what makes the gaps land.
- [ ] **Slide 5**: this is the dense one. Slow down on VV vs VH. Say *"VV is the same polarization going out and coming back. VH is cross-polarized."* This setup is what makes your partner's punchline land in part 2.
- [ ] **Slide 7**: own the line *"we constructed this dataset in this work."* This is your individual contribution to highlight.
- [ ] **Slide 9 handoff**: rehearse the final two sentences most. The colour-coded line on screen and the spoken line should land together. Then yield the floor cleanly.

## Total spoken time

Conservative pace (with pauses): **5 minutes 30 seconds**. If you need to compress to 4 minutes, the easiest cuts:

- Drop the Related Work / DeepSARFlood comparison entirely (it's not currently on a slide, only in the speech).
- Compress slide 8 to "standard fine-tuning, 10 epochs, AdamW, Dice + BCE."
- Skip the acquisition-history note on slide 7.

If you have time to expand to 7 minutes (e.g., for a longer video), add a brief mention on slide 6 that "the OOD failure on Pakistan-2022 is what we'll diagnose in part 2 — and the diagnosis is *not* what you'd expect."
