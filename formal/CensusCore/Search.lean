/-
Executable search for circulant Williamson quadruples of odd order `n` (no Mathlib).

Rows are symmetric `±1` sequences with `x_0 = 1`, coded by `v : List Bool` of length
`H = (n-1)/2` (`v[u-1] = true` iff `x_u = x_{n-u} = -1`).  For each row the program computes the
PAF vector and two linear forms of it: a coarse key (`ck`, two integers) and a fine key (`fk`,
three integers).  For a quadruple the four coarse keys add up to `4n · wc(0)` and the four fine
keys to `4n · wf(0)`, whatever the weights.  The search for one row-sum tuple
`(sA, sB, sC, sD)` groups the rows by coarse key, pairs the groups of `A` and `B` and of `C` and
`D` by the sum of their coarse keys, and inside each matching pair of groups joins the `(A, B)`
row pairs with the `(C, D)` row pairs on the fine key.  It returns the `(C, D)` pairs that
have a partner; an empty list means no quadruple with these row sums exists (proved in
`GoethalsSeidel.Census.Order35`).
-/
import Std.Data.HashMap
import Std.Data.HashSet

namespace CensusCore

open Std

abbrev K2 := Int × Int
abbrev K3 := Int × Int × Int

def add2 (a b : K2) : K2 := (a.1 + b.1, a.2 + b.2)
def sub2 (a b : K2) : K2 := (a.1 - b.1, a.2 - b.2)
def add3 (a b : K3) : K3 := (a.1 + b.1, a.2.1 + b.2.1, a.2.2 + b.2.2)
def sub3 (a b : K3) : K3 := (a.1 - b.1, a.2.1 - b.2.1, a.2.2 - b.2.2)

/-- All Boolean lists of length `k`. -/
def allVecs : Nat → List (List Bool)
  | 0 => [[]]
  | k + 1 => (allVecs k).flatMap (fun v => [false :: v, true :: v])

/-- `Σ_{i<k} f i`. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | k + 1 => sumTo f k + f k

/-- Entry `j` (`0 ≤ j < n`) of the symmetric row with half-code `v`. -/
def entry (n : Nat) (v : List Bool) (j : Nat) : Int :=
  let u := if j ≤ n - j then j else n - j
  if u = 0 then 1 else if v.getD (u - 1) false then -1 else 1

def rowArr (n : Nat) (v : List Bool) : Array Int := Array.ofFn (fun j : Fin n => entry n v j)

def pafAt (n : Nat) (x : Array Int) (k : Nat) : Int :=
  sumTo (fun j => x[j]! * x[(j + k) % n]!) n

def pafArr (n : Nat) (x : Array Int) : Array Int := Array.ofFn (fun k : Fin n => pafAt n x k)

/-- The linear form `Σ_{s<n} w(s) p(s)`. -/
def lin (n : Nat) (w : Nat → Int) (p : Array Int) : Int := sumTo (fun s => w s * p[s]!) n

/-- Weights of the two keys. -/
structure Params where
  n : Nat
  H : Nat
  wc1 : Nat → Int
  wc2 : Nat → Int
  wf1 : Nat → Int
  wf2 : Nat → Int
  wf3 : Nat → Int

structure RowData where
  v : List Bool
  sum : Int
  ck : K2
  fk : K3

def mkRow (P : Params) (v : List Bool) : RowData :=
  let x := rowArr P.n v
  let p := pafArr P.n x
  { v := v
    sum := sumTo (fun j => x[j]!) P.n
    ck := (lin P.n P.wc1 p, lin P.n P.wc2 p)
    fk := (lin P.n P.wf1 p, lin P.n P.wf2 p, lin P.n P.wf3 p) }

/-- The coarse and fine targets: the sums of the four keys of a quadruple. -/
def ctarget (P : Params) : K2 := (4 * P.n * P.wc1 0, 4 * P.n * P.wc2 0)
def ftarget (P : Params) : K3 := (4 * P.n * P.wf1 0, 4 * P.n * P.wf2 0, 4 * P.n * P.wf3 0)

/-- Group a list by a key. -/
def groupBy {α κ : Type} [BEq κ] [Hashable κ] (key : α → κ) (l : List α) :
    HashMap κ (List α) :=
  l.foldl (fun m a => m.insert (key a) (a :: m.getD (key a) [])) {}

/-- The join of the `(A, B)` and `(C, D)` sides.  `T` and `T'` are the coarse and fine
targets. -/
def join (T : K2) (T' : K3) (LA LB LC LD : List RowData) : List (List Bool × List Bool) :=
  let gA := (groupBy (fun r => r.ck) LA).toList
  let gB := (groupBy (fun r => r.ck) LB).toList
  let gC := (groupBy (fun r => r.ck) LC).toList
  let gD := (groupBy (fun r => r.ck) LD).toList
  let abC := gA.flatMap fun ka => gB.map fun kb => (add2 ka.1 kb.1, (ka.2, kb.2))
  let cdC := gC.flatMap fun kc => gD.map fun kd => (sub2 T (add2 kc.1 kd.1), (kc.2, kd.2))
  let abM := groupBy (fun e => e.1) abC
  let cdM := groupBy (fun e => e.1) cdC
  abM.toList.flatMap fun g =>
    let cds := cdM.getD g.1 []
    if cds.isEmpty then [] else
    let S : HashSet K3 := HashSet.ofList
      (g.2.flatMap fun e => e.2.1.flatMap fun a => e.2.2.map fun b => add3 a.fk b.fk)
    cds.flatMap fun e => e.2.1.flatMap fun c =>
      (e.2.2.filter fun d => S.contains (sub3 T' (add3 c.fk d.fk))).map fun d => (c.v, d.v)

def allRows (P : Params) : List RowData := (allVecs P.H).map (mkRow P)

def search (P : Params) (rows : List RowData) (dec : Int × Int × Int × Int) :
    List (List Bool × List Bool) :=
  join (ctarget P) (ftarget P) (rows.filter (fun r => r.sum == dec.1))
    (rows.filter (fun r => r.sum == dec.2.1)) (rows.filter (fun r => r.sum == dec.2.2.1))
    (rows.filter (fun r => r.sum == dec.2.2.2))

def searchAll (P : Params) (decs : List (Int × Int × Int × Int)) : Bool :=
  let rows := allRows P
  decs.all fun dec => (search P rows dec).isEmpty

/-! ### The parameters for `n = 35` -/

/-- Coarse weights: `2048^(s mod 5)` and `2048^(s mod 7)`, i.e. the PAFs of the 7- and
5-compressions. -/
def P35 : Params where
  n := 35
  H := 17
  wc1 s := 2048 ^ (s % 5)
  wc2 s := 2048 ^ (s % 7)
  wf1 s := if 1 ≤ s ∧ s ≤ 6 then 256 ^ (s - 1) else 0
  wf2 s := if 7 ≤ s ∧ s ≤ 12 then 256 ^ (s - 7) else 0
  wf3 s := if 13 ≤ s ∧ s ≤ 17 then 256 ^ (s - 13) else 0

/-- The signed row-sum tuples of `n = 35` up to order (`x_0 = 1` fixes the sign). -/
def decs35 : List (Int × Int × Int × Int) := [(11, -1, 3, 3), (-9, -1, 7, 3), (-9, -5, -5, 3)]

end CensusCore
