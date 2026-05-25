# Speaker notes for the unified slide deck

Total target time: about ten and a half minutes. Two presenters; the handoff is between Slide 9 and Slide 10.

- **Part 1 (Mahbubur): Slides 1 to 9, about five and a half minutes.**
- **Part 2 (Aksan): Slides 10 to 17, about five minutes.**

Each section below is the full speech for that slide, written to be delivered smoothly, in order, without breaks. Read it as one continuous narrative.

---

## PART 1: Mahbubur (Slides 1 to 9)

### Slide 1: Title

Good afternoon. I'm Mahbubur Rahman, and along with my co-author Aksan Gony Alif, we present *Parameter-Efficient Adaptation of SAM 2 for Sentinel-1 SAR Flood Inundation Mapping in the Indo-Gangetic Region*. Our motivation is direct. Bangladesh floods every monsoon, and the monsoon cloud cover means optical satellites can't see the ground exactly when flood maps are most urgently needed. Radar can see through clouds. But radar imagery looks nothing like the natural photographs that modern vision models like SAM 2 were trained on. So the question we set out to answer is whether we can adapt a billion-mask foundation model to noisy, single-channel radar imagery, cheaply, and have it generalize to a flood event the model has never seen before.

### Slide 2: Why this paper exists

The setup, briefly. Bangladesh floods every monsoon, and during monsoon season optical satellites are unreliable precisely because cloud cover blocks the optical sensors. Sentinel-1 SAR is the operational alternative, it penetrates clouds, it's global, and it's free under the Copernicus open data policy. But SAR is fundamentally different from RGB photography. It's a noisy single-channel logarithmic-intensity signal where bright versus dark means rough versus smooth surface, not red versus blue light. Foundation models like SAM 2 weren't trained on anything like it, so they don't transfer naturally. Our research question follows directly from that. Can we adapt SAM 2 to radar, cheaply, and have it generalize to unseen flood events? The short answer is yes, mostly, with one important caveat about the cross-polarization channel that my co-author will get to in part 2.

### Slide 3: Three gaps in prior work

We frame the paper around three concrete gaps in the literature. First, nobody has compared parameter-efficient fine-tuning methods on SAM-family backbones for SAR floods. The literature has scattered single-method papers, one uses LoRA, another uses AdaptFormer, a third uses Convolutional LoRA, but no head-to-head comparison. So a practitioner can't tell which adapter to pick. Second, prior SAR-flood uncertainty work reports low Expected Calibration Error and stops there. Aggregate calibration is fine, but in deployment, operators want per-pixel uncertainty. They want to know which pixels to trust. That's a different question, and nobody had tested it directly. Third, there is no public, reproducible SAR flood test set that is genuinely out-of-distribution by both geography and time. Sen1Floods11 holds out one country, but all its events are from 2017 to 2019. You can't test generalization to a fresh 2022 event without building your own dataset. This paper closes all three gaps.

### Slide 4: Methodology: the empirical sweep

Here is our full evaluation grid. Five backbones, four PEFT methods, three random seeds per cell. The principal pair is SAM ViT-Base at 94 million parameters and SAM 2 Hiera-Base-Plus at 81 million parameters, and these two get the full cross-product of four PEFT methods: LoRA, DoRA, Convolutional LoRA, and AdaptFormer. Then we stretch up to three larger backbones, ViT-Large at 308 million, ViT-Huge at 636 million, and SAM 2 Hiera-Large at 224 million, under Convolutional LoRA only, to test whether the small-backbone findings transport at scale. The total cost across the entire sweep, including the U-Net baseline and the zero-shot baselines, is about three and a half hours on a dual-GPU cloud instance for under ten US dollars. The full sweep is genuinely reproducible at that price. For uncertainty, we use Monte Carlo dropout with twenty stochastic forward passes per chip.

### Slide 5: Polarimetric pseudo-RGB input

Now the SAR-specific piece, which becomes critical in part 2. Sentinel-1 gives us two polarization channels. VV is the co-polarization: vertical-transmit, vertical-receive. VH is cross-polarized: vertical-transmit, horizontal-receive. But SAM 2 was pretrained on three RGB channels, not two SAR channels. So we have to synthesize a third channel from the two we have. This is what we call a pseudo-RGB composition. We compare three of them. The ratio composition puts the log ratio of VV over VH in the third channel. The diff composition uses dB-domain subtraction. The single-polarization control just replicates VV across all three channels and throws VH away. One mathematical note: ratio and diff are mathematically equivalent by a dB-domain identity. They produce numerically identical inputs to the model. So the meaningful contrast is between dual-polarization, where ratio and diff sit, and single-polarization, where VH is removed entirely.

### Slide 6: Four evaluation splits

Our evaluation surface has four splits. Three of them come from Sen1Floods11. The in-distribution test set with ninety chips across ten countries. The Bolivia held-out split with fifteen chips, which is the only truly held-out country in the standard benchmark. And a twelve-chip Pakistan subset from the 2010 event, regionally informative but still in-distribution because Pakistan also appears in the training set. The fourth split, highlighted on screen, is what we built in this work. The Pakistan-2022 set, thirty-nine chips covering the August-September 2022 Indo-Gangetic monsoon flood. This is the same meteorological system that produces Bangladesh's annual flooding, and it is the only strictly out-of-distribution test surface in this evaluation, by both geography and time.

### Slide 7: Pakistan-2022 dataset construction

The Pakistan-2022 dataset is built from two complementary public sources. The labels come from a TU Wien Sentinel-1-derived flood-extent product released alongside the Roth et al. NHESS 2023 paper, under a CC-BY license. The imagery comes from Microsoft Planetary Computer's `sentinel-1-rtc` catalogue: Radiometric Terrain Corrected Sentinel-1 GRD products, fully georeferenced, with anonymous SAS-token access, no account needed. Our pipeline, for each TU Wien mask date, finds the matching Sentinel-1 scene, reads it at native ten-metre resolution, reprojects the mask upward to ten metres, and tiles paired imagery and labels into 512-by-512 chips. The result is thirty-nine chips at native ten-metre resolution, fully Sen1Floods11-compatible in directory layout. One acquisition note we document honestly in the paper: the first chip set was built at twenty metres and mismatched the Sen1Floods11 training distribution. We rebuilt at native ten metres, and that is the dataset released here.

### Slide 8: Experimental setup

A quick word on the training protocol. Each cell trains for ten epochs on the 252 hand-labelled Sen1Floods11 training chips. The optimizer is AdamW with two learning rates: 10 to the minus four for adapter parameters, and 10 to the minus five for any unfrozen pretrained weights. The schedule is cosine decay with a 200-step linear warmup. The loss is Dice plus binary cross-entropy with logits, equally weighted. Mixed precision throughout, effective batch size four. Importantly, we do not apply training-time data augmentation. With only 252 training chips we want every training signal to come from real labels, not synthetic transformations. Test-time augmentation is applied only at inference, only for the ensemble condition reported in part 2. Three random seeds per cell: 42, 123, and 20025.

### Slide 9: Transition to results

Setup complete. The next half of this talk is what we actually found across these four evaluation splits, plus one diagnostic finding that we did not expect, which turned out to be the most operationally useful result in the paper. The cross-polarization channel, the one that helps in-distribution, turns out to be the operative out-of-distribution failure mode. Aksan, take it from here.

---

## PART 2: Aksan (Slides 10 to 17)

### Slide 10: Headline results

Thanks, Mahbubur. Here is the headline results table, and there are three things to take away from it. First, in-distribution. AdaptFormer on Hiera-Base-Plus wins with 0.634 IoU, beating the U-Net baseline at 0.601, at a fraction of the trainable parameter cost. So on the data we trained on, AdaptFormer is the strongest adapter. Second, on the Bolivia held-out split, a different country the model has never seen, DoRA on Hiera-Base-Plus reaches 0.637, essentially matching the U-Net at 0.684, again at far fewer trainable parameters. So far so good. But third, and this is the puzzle, on Pakistan-2022, every SAM-family cell collapses to between 0.09 and 0.14 IoU, while the U-Net retains 0.672. Same training data, same evaluation protocol, but the adapted SAM models fail dramatically on this one new event.

### Slide 11: The collapse is universal

A natural first hypothesis is that one specific backbone or one specific PEFT method is broken on this dataset. We can rule that out completely. The collapse happens on every backbone we tested, ViT-Base, ViT-Large, ViT-Huge, Hiera-Base-Plus, Hiera-Large. It happens on every PEFT method, LoRA, DoRA, Conv-LoRA, AdaptFormer. And it happens consistently across all three random seeds. Meanwhile the U-Net on the same training data does not collapse, which tells us the Pakistan-2022 test set itself is not the problem. So the failure is specific to the adapted SAM-family models. The question becomes: what is the SAM family seeing in this data that the U-Net is not?

### Slide 12: The polarimetric diagnostic

The answer comes from our polarimetric ablation, and it is striking. Same architectures, same training data, same Pakistan-2022 test set. We change just one thing: we drop the cross-polarization VH channel from the input and feed only the co-polarized VV channel, replicated across all three input channels. The Pakistan-2022 IoU jumps dramatically. Convolutional LoRA goes from 0.113 dual-polarization to 0.632 single-polarization. LoRA goes from 0.144 to 0.631. DoRA goes from 0.131 to 0.658. AdaptFormer goes from 0.092 to 0.556. Across three random seeds, the single-polarization Pakistan-2022 IoU lands between 0.56 and 0.66, within 0.01 to 0.12 of the U-Net baseline at 0.672. None of the per-cell confidence intervals overlap with the dual-polarization band. So one channel, drop it, and the collapse disappears.

### Slide 13: Qualitative comparison

The figure on screen shows this qualitatively. Three Pakistan-2022 chips spanning low, mid, and high flood density: 0.2 percent, 6 percent, and 79 percent flood fraction. For each chip, you see the VV intensity image, then the ground-truth flood label, then the U-Net prediction, then the dual-polarization SAM prediction, and then the single-polarization SAM prediction. The dual-polarization SAM under-predicts at every flood-density level. The single-polarization variant, in contrast, reconstructs flood extents that look qualitatively similar to both the U-Net and the ground truth. One channel switched off, and the predictions change from broken to usable.

### Slide 14: Why this happens

So why does removing a channel actually help? The cross-polarization channel carries useful signal in-distribution; that's the trade-off we showed in part 1, where dropping VH cost about 0.08 IoU on the in-distribution test split. But the marginal statistics of that VH channel on the 2022 Pakistan acquisition differ from the 2017-to-2019 Sen1Floods11 training events. Possible drivers are different scattering geometry, different surface conditions, or different acquisition mode parameters between training and Pakistan-2022. Whatever the source, the SAM transformer's global attention amplifies that distribution shift, while the U-Net's local receptive fields tolerate it. Operationally this gives us a concrete recommendation. For new regions or new events, abstain from the cross-polarization channel at deployment, and use VV only. Six-fold improvement, two lines of code.

### Slide 15: Uncertainty: a substantive negative finding

The other contribution worth highlighting is a negative finding about uncertainty, and we report it honestly because the field hasn't really tested this question before. Our Monte Carlo dropout estimator is well-calibrated in the aggregate sense. The Expected Calibration Error is 0.053 on in-distribution test, right at the conventional well-calibrated threshold, and 0.067 on Pakistan-2022, essentially unchanged despite the IoU collapse. But the pointwise selective-prediction curves tell a different story. If we abstain on the highest-uncertainty pixels and re-compute IoU on the remaining ones, the IoU does not improve. The curves are flat. So aggregate calibration does not imply that per-pixel confidences correctly rank pixels by difficulty on out-of-distribution events. This contradicts the implicit assumption in prior SAR-flood uncertainty work, that low aggregate ECE translates into operationally useful per-pixel uncertainty. For our application, pixel-level triage of flood maps, MC dropout's per-pixel scores are not the right tool. Sharper estimators like deep ensembles or learned reject heads are needed, and we flag that as future work.

### Slide 16: Three gaps, three closures

To recap the contributions against the three gaps we opened with. First, the PEFT comparison. We document a real accuracy-versus-generalization trade-off. AdaptFormer maximizes in-distribution IoU, but the LoRA family is the more generalization-robust choice on held-out countries. Second, the calibration question. We surface a substantive negative finding: aggregate ECE does not imply per-pixel uncertainty utility on OOD. Third, the public OOD test set. We release Pakistan-2022, thirty-nine chips at native ten-metre resolution, with the full acquisition pipeline. And as a bonus diagnostic finding, the cross-polarization channel turns out to be the operative OOD failure mode for SAM-family adapters, which gives us a concrete deployment recommendation: use VV only for new regions.

### Slide 17: Limitations, future work, and close

A few honest limitations. Pakistan-2022 rests on thirty-nine chips, and the labels themselves were derived from a Sentinel-1 segmentation pipeline rather than from independent ground truth, so the U-Net's matching IoU partly reflects methodological similarity, not just CNN robustness. The stretch backbones are evaluated only under Convolutional LoRA, not the full PEFT cross-product. And the Bangladesh-Sylhet construction pipeline is released alongside the codebase, but its chips are not yet built or empirically evaluated. Future work is clear from these. Empirically evaluate on the Bangladesh-Sylhet set. Characterize the cross-polarization distribution shift directly through acquisition metadata like incidence angle and acquisition mode. And explore sharper uncertainty estimators, deep ensembles and learned reject heads, that could fix the per-pixel utility gap. Thank you. We're happy to take questions.

---

## Quick delivery checklist

- On Slide 1, say both names slowly and look at the camera, not at the slide.
- On Slide 3, leave a short beat between *First*, *Second*, and *Third* so each gap registers.
- On Slide 5, slow down on VV and VH, because this is the setup that makes the punchline on Slide 12 land.
- On Slide 7, own the line about "we constructed this dataset in this work", it's an individual contribution worth highlighting.
- On the Slide 9 to Slide 10 handoff, leave a clean beat of silence between Mahbubur's final line and Aksan's opening "Thanks, Mahbubur." Avoid talking over each other.
- On Slide 12, the numbers (0.113 to 0.632, 0.144 to 0.631, 0.131 to 0.658, 0.092 to 0.556) are the punchline of the entire talk. Rehearse them until they land confidently.
- End Slide 17 on the line "Thank you. We're happy to take questions." Don't let the talk peter out at the limitations.

## Pacing reference

At a normal conversational pace of about 150 words per minute, the full speech above runs to approximately ten minutes and thirty seconds. If you need to compress to eight minutes, the cleanest cuts are Slide 11 (the universality of the collapse is already visible from the table on Slide 10) and Slide 14 (the "why this happens" can be implied by the result on Slide 12).
