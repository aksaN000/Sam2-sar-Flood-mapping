# Video presentation slides

A single unified LaTeX Beamer deck for the 10-minute video presentation accompanying the paper. Two-presenter format with the handoff at Slide 9 / Slide 10.

## Files

| File | Contents |
|---|---|
| `slides.tex` | The unified deck. Slides 1–9: Mahbubur (motivation, methods, datasets, setup, transition). Slides 10–17: Aksan (results, polari diagnostic, uncertainty, close). |
| `slides.pdf` | Compiled output (17 slides, 16:9). |
| `speaker_notes.md` | Per-slide speaking script for both presenters + delivery checklist. |

## How to compile

```bash
cd report/video_slides
pdflatex slides.tex
```

Uses the `metropolis` Beamer theme. If `metropolis` is missing on your TeX install, replace `\usetheme{metropolis}` with `\usetheme{default}` or `\usetheme{Singapore}`.

## Recording tips

- **Aspect ratio:** 16:9 (set via `\documentclass[aspectratio=169,11pt]{beamer}`).
- **Font size:** 11 pt base, readable on any video platform.
- **Colours:** accent blue (`#1E64C8`) for in-paper-contribution highlights; warning red (`#C81E1E`) for the Pakistan-2022 collapse and cross-polarization punchline.
- **Pacing:** Mahbubur ~5:30, Aksan ~5:00. Total slot: ~10:30.
- **Handoff:** Slide 9 is Mahbubur's transition slide. Slide 10 picks up with the headline results. The natural pause for the speaker switch is between them. Speaker notes script the exact handoff line.

## Visual assets

Slide 13 embeds `../figs/pakistan2022_overlay.pdf` from the paper, the qualitative 3-chip comparison that visualizes the cross-polarization finding. All other slides use text + simple tables so the deck stays small and readable on video.

If you want to add more figures, they live in `report/figs/`: `iou_vs_bolivia.pdf`, `ood_gap.pdf`, `reliability_diagram.pdf`, `selective_prediction.pdf`, `iou_per_million_params.pdf`.

## Rehearsal flow

1. Compile the deck.
2. Both presenters read `speaker_notes.md` end-to-end.
3. Mahbubur rehearses Slides 1–9 with a timer; aim for 5:30.
4. Practice the handoff (Slide 9 last line → Slide 10 opening line).
5. Aksan rehearses Slides 10–17; aim for 5:00.
6. One full rehearsal end-to-end before recording.
