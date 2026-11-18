/-
The G-matrix quadruple of order 49 on the board (entry n49-703082c3c1ffa487f376), checked by
kernel evaluation. A published exhaustive search reports that none exists at this order.
-/
import GoethalsSeidel.Basic

namespace GoethalsSeidel.Order49

/-- The sequence with the given list of entries, `0` outside the list. -/
def ofList (l : List ℤ) (j : ZMod 49) : ℤ := l.getD j.val 0

def rowA : List ℤ :=
  [1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1]

def rowB : List ℤ :=
  [1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1]

def rowC : List ℤ :=
  [1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1]

def rowD : List ℤ :=
  [1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1]

def a := ofList rowA
def b := ofList rowB
def c := ofList rowC
def d := ofList rowD

set_option maxRecDepth 100000 in
theorem isQuadruple : IsQuadruple a b c d where
  signA := by unfold IsSign; decide +kernel
  signB := by unfold IsSign; decide +kernel
  signC := by unfold IsSign; decide +kernel
  signD := by unfold IsSign; decide +kernel
  paf_zero := by unfold paf; decide +kernel

set_option maxRecDepth 100000 in
theorem isSkew_a : IsSkew a := by unfold IsSkew; decide +kernel

set_option maxRecDepth 100000 in
theorem isSkew_b : IsSkew b := by unfold IsSkew; decide +kernel

set_option maxRecDepth 100000 in
theorem isSymmetric_c : IsSymmetric c := by unfold IsSymmetric; decide +kernel

set_option maxRecDepth 100000 in
theorem isSymmetric_d : IsSymmetric d := by unfold IsSymmetric; decide +kernel

/-- G-matrices of order 49 exist: a quadruple with two skew and two symmetric blocks. -/
theorem exists_g_quadruple :
    ∃ a b c d : ZMod 49 → ℤ,
      IsQuadruple a b c d ∧ IsSkew a ∧ IsSkew b ∧ IsSymmetric c ∧ IsSymmetric d :=
  ⟨a, b, c, d, isQuadruple, isSkew_a, isSkew_b, isSymmetric_c, isSymmetric_d⟩

end GoethalsSeidel.Order49
