# Speaker notes for the unified slide deck

Total target time: **~10:30**. Two presenters; the handoff is between Slide 9 and Slide 10.

- **Part 1 (Mahbubur): Slides 1–9, ~5:30.** Title, motivation, three gaps, methodology, polarimetric input, datasets, experimental setup, transition.
- **Part 2 (Aksan): Slides 10–17, ~5:00.** Headline results, the Pakistan-2022 collapse, the polarimetric diagnostic, qualitative figure, why-it-happens, uncertainty negative finding, three-gap closure, limitations + close.

Pace: ~150 words/minute, conversational. Each section below is what to *say*, not what to read off the screen.

---

## PART 1: Mahbubur

### Slide 1: Title (~30 sec)

> Good [morning / afternoon]. I'm Mahbubur Rahman, and along with my co-author Aksan Gony Alif, we present *Parameter-Efficient Adaptation of SAM 2 for Sentinel-1 SAR Flood Inundation Mapping in the Indo-Gangetic Region*.
>
> The motivation is direct. Bangladesh floods every monsoon. Optical satellites can't see through monsoon clouds, exactly when flood maps matter most. Radar can. But radar imagery looks nothing like the natural photographs that modern vision models like SAM 2 were trained on. Our question is: can we adapt a billion-mask foundation model to noisy single-channel radar, cheaply, and have it actually generalize to a flood event it has never seen?

### Slide 2: Why this paper exists (~45 sec)

> Bangladesh floods every monsoon. Optical satellites are unreliable precisely when flood mapping is most urgent, because monsoon clouds block the optical sensors. Sentinel-1 SAR penetrates clouds, is global, and is free.
>
> But SAR is fundamentally different from RGB photography. It's a noisy single-channel logarithmic-intensity signal where bright versus dark means rough versus smooth surface, not red versus blue light. Foundation models like SAM 2 don't transfer to it naturally.
>
> Our research question follows. Can we adapt SAM 2 to radar cheaply, and have it generalize to unseen flood events? The answer, mostly yes, with one important caveat about the cross-polarization channel that Aksan will get to in part 2.

### Slide 3: Three gaps in prior work (~50 sec)

> We frame the paper around three concrete gaps in the literature.
>
> *Pause briefly before each gap.*
>
> First, nobody has compared parameter-efficient fine-tuning methods on SAM-family backbones for SAR floods. The literature has scattered single-method papers, one uses LoRA, another uses AdaptFormer, but no head-to-head comparison. A practitioner can't tell which adapter to pick.
>
> Second, prior SAR-flood uncertainty work reports low Expected Calibration Error and stops there. Aggregate calibration is fine, but in deployment operators want per-pixel uncertainty, *which pixels should I trust*. That's a different question, and nobody tests it directly.
>
> Third, there is no public, reproducible SAR flood test set that is genuinely out-of-distribution by both *geography and time*. Sen1Floods11 holds out one country but all its events are from 2017 to 2019. You can't test generalization to a fresh 2022 event without building your own dataset.
>
> This paper closes all three.

### Slide 4: Methodology: the sweep (~50 sec)

> Here is our full evaluation grid. Five backbones, four PEFT methods, three random seeds per cell.
>
> The principal pair is SAM ViT-Base at 94 million parameters and SAM 2 Hiera-Base-Plus at 81 million parameters. These two get the full cross-product: LoRA, DoRA, Convolutional LoRA, and AdaptFormer.
>
> Then we stretch up to three larger backbones, ViT-Large, ViT-Huge at 636 million, and SAM 2 Hiera-Large, under Convolutional LoRA only, to test whether the small-backbone findings transport at scale.
>
> Total cost across the entire sweep, including the U-Net baseline and the zero-shot baselines, is about three and a half hours on a dual-GPU cloud instance for under ten US dollars. The full sweep is reproducible at that price.
>
> Uncertainty is added via Monte Carlo dropout with 20 stochastic forward passes per chip.

### Slide 5: Polarimetric pseudo-RGB input (~50 sec)

> Now the SAR-specific bit. Sentinel-1 gives us two polarization channels. *VV*, vertical-transmit, vertical-receive, the co-polarization. And *VH*, the cross-polarized channel.
>
> But SAM 2 was pretrained on three RGB channels. So we have to synthesize a third channel from the two we have. This is what we call a "pseudo-RGB" composition.
>
> We compare three of them. The *ratio* composition puts the log ratio of VV over VH in the third channel. The *diff* composition uses dB-domain subtraction. The *single-polarization* control just replicates VV across all three channels and throws VH away.
>
> A small mathematical note: ratio and diff are mathematically equivalent by a dB-domain identity. They produce numerically identical inputs to the model. So the meaningful contrast, and this becomes critical in part 2 of the talk, is between *dual-polarization* and *single-polarization*.

### Slide 6: Four evaluation splits (~45 sec)

> Our evaluation surface has four splits.
>
> Three come from Sen1Floods11. The in-distribution test set with 90 chips across ten countries. The Bolivia held-out split with 15 chips, the only truly held-out country in the standard benchmark. And a 12-chip Pakistan subset from the 2010 event, regionally informative but still in-distribution.
>
> The fourth split, *highlighted on screen*, is what we built in this work. The Pakistan-2022 set. 39 chips covering the August-September 2022 Indo-Gangetic monsoon flood. This is the same meteorological system that produces Bangladesh's annual flooding, and it is the only strictly out-of-distribution test surface in this evaluation, by both *geography and time*.

### Slide 7: Pakistan-2022 dataset construction (~50 sec)

> The Pakistan-2022 dataset is built from two complementary public sources.
>
> Labels come from a TU Wien Sentinel-1-derived flood-extent product released alongside the Roth et al. NHESS 2023 paper, under a CC-BY license.
>
> Imagery comes from Microsoft Planetary Computer's `sentinel-1-rtc` catalogue, Radiometric Terrain Corrected Sentinel-1 GRD products, fully georeferenced, anonymous SAS-token access. No account needed.
>
> Our pipeline, for each TU Wien mask date: find the matching Sentinel-1 scene, read at native 10 metre resolution, reproject the mask upward to 10 metres, tile into 512 by 512 chips. Result: 39 chips at native 10 metre resolution, fully Sen1Floods11-compatible layout.
>
> One acquisition note we document honestly in the paper: the first chip set was at 20 metres and mismatched the Sen1Floods11 training distribution. We rebuilt at native 10 metres. That's the dataset released in this paper.

### Slide 8: Experimental setup (~30 sec)

> Quick training protocol. Each cell is 10 epochs on the 252 hand-labelled Sen1Floods11 training chips. AdamW optimizer, two learning rates, 10 to the minus four for adapter parameters, 10 to the minus five for any unfrozen pretrained weights. Cosine schedule with 200-step linear warmup. Loss is Dice plus binary cross-entropy, equal weight. Mixed precision throughout. Effective batch size 4.
>
> Importantly, we do *not* apply training-time data augmentation. With only 252 chips, we want every training signal to come from the actual labels. Test-time augmentation is used only at inference, only for the ensemble condition.
>
> Three random seeds per cell: 42, 123, 20025.

### Slide 9: Transition to results (~20 sec)

> *Pause briefly. Look up at camera.*
>
> Setup complete. The next half of this talk is what we actually found across these four evaluation splits, plus one diagnostic finding we did not expect, which turned out to be the most operationally useful result in the paper.
>
> *Pause. Land the line.*
>
> The cross-polarization channel, the one that helps in-distribution, turns out to be the *operative* out-of-distribution failure mode.
>
> Aksan, take it from here.

---

## PART 2: Aksan

### Slide 10: Headline results (~50 sec)

> Thanks, Mahbubur. Here is the headline results table.
>
> Four rows, four splits, with the U-Net baseline at the top for reference.
>
> Three things to take away.
>
> First, *in-distribution*: AdaptFormer on Hiera-Base-Plus wins with 0.634 IoU, beating the U-Net baseline (0.601) at a fraction of the trainable parameters.
>
> Second, on the *Bolivia held-out* split, a different country the model has never seen, DoRA on Hiera-Base-Plus reaches 0.637, essentially matching the U-Net at 0.684 with far fewer trainable parameters.
>
> Third, and this is the puzzle, on *Pakistan-2022*, every SAM-family cell collapses to between 0.09 and 0.14 IoU, while the U-Net retains 0.672. Same training data, same evaluation, but the adapted SAM models fail dramatically on this new event.

### Slide 11: The collapse is universal (~40 sec)

> A natural first hypothesis is that one specific backbone or one specific PEFT method is broken. We can rule that out.
>
> The collapse happens on every backbone we tested, ViT-Base, ViT-Large, ViT-Huge, Hiera-Base-Plus, Hiera-Large. It happens on every PEFT method, LoRA, DoRA, Conv-LoRA, AdaptFormer. And it happens consistently across three random seeds.
>
> Meanwhile the U-Net on the same training data does *not* collapse. So the Pakistan-2022 test set itself is not the problem. The failure is specific to the adapted SAM-family models.
>
> What is the SAM family seeing that the U-Net is not?

### Slide 12: The polarimetric ablation (~55 sec)

> The answer comes from our polarimetric ablation.
>
> Same architectures, same training data, same Pakistan-2022 test set, but we vary one thing: drop the cross-polarization VH channel from the input. Use only the co-polarized VV channel, replicated.
>
> The Pakistan-2022 IoU jumps dramatically. Conv-LoRA goes from 0.113 to 0.632. LoRA from 0.144 to 0.631. DoRA from 0.131 to 0.658. AdaptFormer from 0.092 to 0.556.
>
> Across three random seeds, single-polarization Pakistan-2022 IoU lands between 0.56 and 0.66, within 0.01 to 0.12 of the U-Net baseline. None of the per-cell confidence intervals overlap with the dual-polarization band.
>
> Drop the cross-polarization channel, and the collapse vanishes.

### Slide 13: Qualitative comparison (~30 sec)

> The figure on screen shows this qualitatively. Three Pakistan-2022 chips spanning low, mid, and high flood density.
>
> For each chip, you see: the VV intensity image, the ground-truth label, the U-Net prediction, the dual-polarization SAM prediction, and the single-polarization SAM prediction.
>
> The dual-polarization SAM under-predicts at every flood-density level. The single-polarization variant reconstructs flood extents that look qualitatively similar to the U-Net and the ground truth.
>
> One channel switched off, and the predictions change from broken to usable.

### Slide 14: Why this happens (~40 sec)

> So why does removing a channel *help*?
>
> The cross-polarization channel carries useful signal in-distribution. But its marginal statistics on the 2022 Pakistan acquisition differ from the 2017–2019 Sen1Floods11 training events. Possible drivers: different scattering geometry, different surface conditions, different acquisition mode parameters between the training distribution and August-September 2022 over Pakistan.
>
> The SAM transformer's global attention amplifies that distribution shift. The U-Net's local receptive fields tolerate it.
>
> Operationally, this gives us a concrete recommendation. For new regions or events, abstain from the cross-polarization channel at deployment. Use VV only. Six-fold improvement, two lines of code.

### Slide 15: Uncertainty: a substantive negative finding (~45 sec)

> The other contribution worth highlighting is a negative finding about uncertainty.
>
> Our Monte Carlo dropout estimator is well-calibrated in the aggregate sense: ECE is 0.053 on in-distribution and 0.067 on Pakistan-2022, essentially unchanged despite the IoU collapse.
>
> But the *pointwise* selective-prediction curves are flat. If we abstain on the highest-uncertainty pixels, the IoU on the remaining pixels does not improve.
>
> Aggregate calibration does not imply that per-pixel confidences rank pixels by difficulty on out-of-distribution events. This contradicts the implicit assumption in prior SAR-flood uncertainty work that low aggregate ECE translates into operationally useful per-pixel uncertainty.
>
> For our application, pixel-level triage of flood maps, MC dropout's per-pixel scores are not the right tool. Sharper estimators like deep ensembles or learned reject heads are needed.

### Slide 16: Three gaps, three closures (~35 sec)

> To recap the contributions against the three gaps we opened with.
>
> First, the PEFT comparison. We document a real accuracy-versus-generalization trade-off. AdaptFormer maximizes in-distribution IoU; the LoRA family is the more generalization-robust choice on held-out countries.
>
> Second, the calibration question. We surface a substantive negative finding: aggregate ECE does not imply per-pixel uncertainty utility on OOD.
>
> Third, the public OOD test set. We release Pakistan-2022, 39 chips at native 10 metre resolution, with the acquisition pipeline.
>
> And as a bonus diagnostic: cross-polarization is the operative OOD failure mode for SAM-family adapters.

### Slide 17: Limitations, future work, close (~35 sec)

> A few honest limitations. Pakistan-2022 rests on 39 chips and on TU Wien labels themselves derived from a Sentinel-1 model, the U-Net's matching IoU partly reflects methodological similarity, not just CNN robustness. The stretch backbones are evaluated only under Conv-LoRA, not the full PEFT cross-product. And the Bangladesh-Sylhet construction pipeline is released alongside the codebase, but its chips are not yet built or evaluated.
>
> Future work is clear: empirically evaluate on the Bangladesh-Sylhet set, characterize the cross-polarization shift directly through acquisition metadata, and explore sharper uncertainty estimators.
>
> Thank you. We're happy to take questions.

---

## Delivery checklist

- **Slide 1**: state both names slowly. Look up at the camera, not at the slide.
- **Slide 3**: pause briefly between *First*, *Second*, *Third*. The three-beat rhythm makes the gaps land.
- **Slide 5**: this is the dense one. Slow down on VV vs VH. Say *"VV is the same polarization going out and coming back. VH is cross-polarized."* This sets up the partner's punchline.
- **Slide 7**: own the line *"we constructed this dataset in this work."* Individual contribution to highlight.
- **Slide 9 → Slide 10 handoff**: rehearse this transition most. Mahbubur finishes; brief silence; Aksan picks up cleanly with "Thanks, Mahbubur." Avoid talking over each other.
- **Slide 12**: the numbers (0.113 → 0.632, etc.) need to land confidently. This is the punchline of the entire talk.
- **Slide 17**: end on the line *"Thank you. We're happy to take questions."* Don't let the talk peter out.

## Total spoken time

Conservative pace with pauses: **~10:30**. If you need to compress to 8:00, the easiest cuts are Slide 11 (collapse universality, visually clear from the table on Slide 10) and Slide 14 (the "why", can be implicit from the result on Slide 12).
