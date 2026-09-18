"""Run GRAFT on an image and save the result.

    # you already have a photo of the empty scene
    python demo.py photo.jpg --scene-image empty_room.jpg -o out/

    # you don't -- segment and inpaint the people out to make one
    python demo.py photo.jpg --inpaint-scene -o out/

Writes, per input image:

    <name>.glb            bodies + scene in one glTF file, for any viewer or browser
    <name>_bodies.obj     the posed SMPL-X meshes, in the scene's camera frame
    <name>_scene.ply      the scene point cloud GRAFT fitted against
    <name>_smplx.npz      the SMPL-X parameters, (n_people, ...) arrays
    <name>_person_mask.png the person mask GRAFT aligned the two views with
    <name>_scene_used.png the empty-scene view that was used (with --inpaint-scene)

The two meshes share one coordinate frame, so loading both into any viewer shows
the bodies sitting in the scene.
"""
from __future__ import annotations

import os
from pathlib import Path

import click


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--scene-image", type=click.Path(exists=True), default=None,
              help="A view of the same scene without the people. Strongly recommended: "
                   "this is where the geometry comes from.")
@click.option("--inpaint-scene", is_flag=True,
              help="Synthesise the empty-scene view by segmenting and inpainting the "
                   "people out.")
@click.option("--refine-hands", is_flag=True,
              help="Refine hand poses with WiLoR (needs the 'demo' extra).")
@click.option("--model", "model_name", default="graft", show_default=True,
              help="A name from graft.PRETRAINED, or a path to a checkpoint.")
@click.option("--num-iters", default=3, show_default=True, help="Refinement iterations.")
@click.option("-o", "--out-dir", type=click.Path(), default="outputs", show_default=True)
@click.option("--data-root", type=click.Path(), default=None, help="Overrides $GRAFT_DATA_ROOT.")
@click.option("--device", default="cuda", show_default=True)
def main(images, scene_image, inpaint_scene, refine_hands, model_name,
         num_iters, out_dir, data_root, device):
    if data_root:
        os.environ["GRAFT_DATA_ROOT"] = str(Path(data_root).expanduser().resolve())

    import numpy as np
    import trimesh
    from PIL import Image

    from graft import GRAFTModel
    from graft.export import write_glb
    from graft.scene import person_mask

    if scene_image and inpaint_scene:
        raise SystemExit("Pass either --scene-image or --inpaint-scene, not both.")
    if not scene_image and not inpaint_scene:
        click.echo(
            "note: no empty-scene view given, so the input image is used as its own "
            "scene view. The people then occlude the geometry being fitted; pass "
            "--scene-image or --inpaint-scene for the intended setting.", err=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = GRAFTModel.from_pretrained(model_name).to(device).eval()
    faces = model.faces

    for path in images:
        path = Path(path)
        stem = path.stem
        scene = scene_image
        # The person view is aligned onto the scene view on the pixels outside this
        # mask, so it is needed whichever way the scene view is obtained.
        click.echo(f"{stem}: segmenting the people…")
        mask = person_mask(path)
        mask.save(out / f"{stem}_person_mask.png")
        if inpaint_scene:
            from graft.scene import remove_people
            click.echo(f"{stem}: inpainting people out to build the scene view…")
            scene = remove_people(path, mask, device=device)
            scene.save(out / f"{stem}_scene_used.png")

        click.echo(f"{stem}: running GRAFT…")
        result = model.predict(path, scene_image=scene, person_mask=mask,
                               num_iters=num_iters, refine_hands=refine_hands)
        if len(result) == 0:
            click.echo(f"{stem}: no people detected, skipping.", err=True)
            continue

        bodies = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=v.cpu().numpy(), faces=faces, process=False)
            for v in result.vertices
        ])
        bodies.export(out / f"{stem}_bodies.obj")

        pts = result.points.cpu().numpy().reshape(-1, 3)
        colors = np.asarray(Image.open(scene if scene is not None else path)
                            .convert("RGB").resize(result.points.shape[1::-1]))
        keep = np.isfinite(pts).all(-1)
        trimesh.PointCloud(pts[keep], colors=colors.reshape(-1, 3)[keep]).export(
            out / f"{stem}_scene.ply")

        write_glb(out / f"{stem}.glb", result.vertices.cpu().numpy(), faces,
                  result.points.cpu().numpy(), scene if scene is not None else path)

        np.savez(out / f"{stem}_smplx.npz",
                 **{k: v.cpu().numpy() for k, v in result.smplx.items()},
                 K=result.K.cpu().numpy())
        click.echo(f"{stem}: {len(result)} person(s) -> {out}/{stem}_*")


if __name__ == "__main__":
    main()
