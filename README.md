# convert quip to training data

Convert human dotting labels on QuIP to training data for cell classification task.

## example

```bash
python convert_quip_to_training_data.py \
    --dump-path '/data/quip_distro/data/dump/TCGA:paad-2021-8-14-17-9-47' \
    --svs-root /data/quip_distro/images/tcga_data/paad/ \
    --polygon-root /data/quip_distro/data/kaczmarj/hou-scientific-data-nuclei-v0/paad_polygon/ \
    --roi-name "Waqas 500p Non-Tumor 500p" \
    --label-regex Waqas \
    --subject-ids TCGA-2J-AABT \
    --save-polygon-overlays \
    --output-dir outputs
```

## convert output tiles to a movie

In the following example, we assume that the conversion script put its outputs in
a directory `outputs` and there is a class of labels called `lymphocytes`.

The `convert` command comes from ImageMagick, and to save to mp4, one must install
`ffmpeg`.

```bash
convert -delay 20 -loop 0 outputs/lymphocytes/*.png lymphocytes.mp4
```
