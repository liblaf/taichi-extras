# Deformation Transfer

## Results

The order of the images from left to right is `s0`, `s1`, `t0`, `t1`, `u0` (the new `t0`), and `u1` (the new `t1`).

![Results](https://cdn.liblaf.me/github.com/tiny-mesh-viewer/demo/face.png)

## Algorithm

As explained in detail in Sec. 3 of [^1], Deformation Transfer aims to minimize the distance between the deformation matrix of the source and target meshes. The original research used least squares to solve the minimization problem and introduced an auxiliary vector $\vb{v}_4$. However, I implemented the use of gradient descent instead.

When the inputs are not aligned, the meshes must first be aligned before performing Deformation Transfer. Assume that the pose change before and after alignment can be represented by a transformation matrix $M$ and a translation vector $\vb*{d}$. (Note that $M$ should only contain 4 degrees of freedom for isotropic scaling and rotation, plus 3 degrees of freedom for translation. However, for convenience, my implementation does not place restrictions on $M$. In fact, the results are satisfactory even if $M$ has no restrictions.) Given the source mesh before alignment (denoted as $s_0$) and the target mesh $t_0$, we can derive the alignment matrix by minimizing the distance between $s_0$ and $t_0'$ transformed by $M$ and the translation vector $\vb*{b}$. After transferring the features of $s_1$ to $t_1'$, the desired target $t_1$ can be obtained by performing the inverse transformation of the alignment step.

[^1]: Deformation Transfer for Triangle Meshes <https://doi.org/10.1145/1015706.1015736>
