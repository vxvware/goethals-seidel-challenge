# Linear solver headers

`hkt_solve.h` and `hkt_solve64.h` implement the mod-4 linearization of Holzmann, Kharaghani and Tayfeh-Rezaie used by the G-matrix, good-matrix and Williamson census engines. Both files are pinned by hash in registered census replays, so their text is not edited; corrections to their comments are recorded here.

The header comment in both files says the solver is given "the skew row A and one symmetric row B" and solves for the other two symmetric rows. That describes the good-matrix engines. The Williamson engines (`wil_hkt*.c`) pass two symmetric rows, and the G-matrix engines (`gmat_c*.c`) pass the two skew rows; in every case the solver finds the remaining two rows. Stage 1 fixes, for each index u, whether the two unknown rows agree or differ at u (the comment's "c_u + d_u = q_u (mod 4)"); stage 2 solves for the first unknown row over GF(2).

The mathematics of both stages, in the form the code implements it, is proved in Lean in `formal/`.
