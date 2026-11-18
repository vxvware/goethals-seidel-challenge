/-
The skew quadruple of order 119 on the board (entry n119-f302cc45a2f83e60917b), checked by
kernel evaluation.
-/
import GoethalsSeidel.Basic

namespace GoethalsSeidel.Order119

/-- The sequence with the given list of entries, `0` outside the list. -/
def ofList (l : List ℤ) (j : ZMod 119) : ℤ := l.getD j.val 0

def rowA : List ℤ :=
  [1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1]

def rowB : List ℤ :=
  [1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1]

def rowC : List ℤ :=
  [1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1]

def rowD : List ℤ :=
  [1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1]

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
theorem isSkew : IsSkew d := by unfold IsSkew; decide +kernel

/-- A Goethals–Seidel quadruple of order 119 with a skew block exists. -/
theorem exists_skew_quadruple :
    ∃ a b c d : ZMod 119 → ℤ, IsQuadruple a b c d ∧ IsSkew d :=
  ⟨a, b, c, d, isQuadruple, isSkew⟩

end GoethalsSeidel.Order119
