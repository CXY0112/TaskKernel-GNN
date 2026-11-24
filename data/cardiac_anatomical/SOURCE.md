[中文版](SOURCE.zh-CN.md)

# D2-D5a anatomical mesh source

- Dataset: **A statistical shape model of the human heart: biventricular model v2**
- Authors: Steffen Schuler and Axel Loewe
- Record: <https://zenodo.org/records/4506463>
- Input file: `heart_sur.vtp`
- Download: <https://zenodo.org/records/4506463/files/heart_sur.vtp?download=1>
- MD5: `11069c64928db9faf60d34ffe916b73d`
- License: CC BY 4.0

The source is a closed average biventricular surface with point labels for the
base, epicardium, LV endocardium, RV endocardium, and apex. D2-D5a remeshes the
same surface at several resolutions with Gmsh 4.15.2. This holds anatomy fixed
during Mesh-OOD evaluation.

The experiment uses a documented rule-based transmural helix field because the
surface release does not include subject-specific fibres. Fibres rotate from
+60 degrees at the nearest endocardium to -60 degrees at the epicardium. This is
an anatomical-geometry validation bridge, not a clinical fibre reconstruction.

