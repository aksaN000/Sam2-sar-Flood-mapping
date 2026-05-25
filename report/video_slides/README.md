# Video presentation slides

LaTeX Beamer slides for the 10-minute video presentation accompanying the paper. Two-presenter format.

## Files

| File | Owner | What it covers |
|---|---|---|
| `slides_part1.tex` | Aksan | Title, motivation, three gaps, methodology, polarimetric input, datasets, experimental setup, handoff. ~5–6 min. |
| `slides_part2_skeleton.tex` | Mahbubur | Results (headline + polari diagnostic), uncertainty negative finding, conclusion. ~5 min. Skeleton; fill in delivery details during rehearsal. |
| `speaker_notes_part1.md` | Aksan | Per-slide speaking script + delivery checklist. |

## How to compile

Both decks use the `metropolis` Beamer theme.

```bash
cd report/video_slides
pdflatex slides_part1.tex
pdflatex slides_part2_skeleton.tex
```

Run `pdflatex` twice on each if you add `\ref` or `\cite` calls. The current decks don't need a second pass.

If `metropolis` is missing on your TeX install, replace `\usetheme{metropolis}` with `\usetheme{default}` or `\usetheme{Singapore}` — content will be unchanged, only visual styling differs.

## Recording tips

- **Aspect ratio:** 16:9 (already set via `\documentclass[aspectratio=169,11pt]{beamer}`).
- **Font size:** 11 pt base — readable on any video platform.
- **Colours:** the accent blue (`#1E64C8`) is used for in-paper-contribution highlights; the warning red (`#C81E1E`) is used for the Pakistan-2022 collapse / cross-polarization punchline.
- **Pacing:** Aksan ~5:30, Mahbubur ~5:00. Total slot: ~10:30.
- **The `\note{}` mechanism** is not used here — speaker notes live in `speaker_notes_part1.md` as Markdown for easier rehearsal.

## Visual assets used

The Part 2 slides embed `../figs/pakistan2022_overlay.pdf` from the paper. All other slides use only text + simple tables to keep file size small and readability high on video.

If you want to add more figures from the paper (e.g., `iou_vs_bolivia.pdf`, `ood_gap.pdf`, `reliability_diagram.pdf`), they live in `report/figs/`.

## Suggested rehearsal flow

1. Compile both decks.
2. Read `speaker_notes_part1.md` end-to-end.
3. Aksan rehearses Part 1 with a timer; aim for 5:30.
4. Practice the handoff (Slide 9 of Part 1 → Slide 1 of Part 2).
5. Mahbubur rehearses Part 2; aim for 5:00.
6. One full rehearsal end-to-end before recording.
