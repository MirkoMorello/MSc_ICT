# Figure sources

All three README figures come from
`Final_Project/Morello_Yachaya_Presentation.pdf` (pages counted from 1).
The original PDF is unchanged.

| File | Source | Preparation |
| --- | --- | --- |
| `hardware_renders.png` | Page 4, three embedded images | Extract originals with `pdfimages`, then arrange horizontally with 16 px margins; no image content cropped or upscaled. |
| `hardware_exploded.png` | Page 5, embedded CAD sheet | Extract the complete 1545 × 1191 image directly; no cropping or resampling. |
| `architecture.png` | Page 3, vector diagram | Render at 3000 px page width and crop the full diagram, including the person, speaker, arrows and both labelled lanes. |

To reproduce, run from the repository root with Poppler and ImageMagick.
Temporary files go into a newly created directory:

```sh
figure_tmp=$(mktemp -d)
pdfimages -f 4 -l 4 -png Final_Project/Morello_Yachaya_Presentation.pdf "$figure_tmp/render"
magick montage "$figure_tmp/render-000.png" "$figure_tmp/render-001.png" "$figure_tmp/render-002.png" -tile 3x1 -geometry +16+16 -background '#f4f4f4' docs/figures/hardware_renders.png
pdfimages -f 5 -l 5 -png Final_Project/Morello_Yachaya_Presentation.pdf "$figure_tmp/cad"
cp "$figure_tmp/cad-000.png" docs/figures/hardware_exploded.png
pdftoppm -f 3 -l 3 -scale-to 3000 -x 817 -y 877 -W 2123 -H 750 -singlefile -png Final_Project/Morello_Yachaya_Presentation.pdf "$figure_tmp/architecture"
cp "$figure_tmp/architecture.png" docs/figures/architecture.png
```

ImageMagick may require a locally installed font even without labels;
on macOS add `-font /System/Library/Fonts/Helvetica.ttc` after `montage`.

Keep these filenames stable. The main README pins the raw image URLs to
the image commit because GitHub's branch-image cache continued serving the
old crops even with a query parameter. When regenerating, commit the images
first and update those URLs to the new image commit. Inspect all PNGs at full size and in
the rendered README to ensure no device parts, labels or arrows are cut off.
