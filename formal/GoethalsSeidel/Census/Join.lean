/-
Completeness of the hash join in `CensusCore.join`: if four rows from the four lists have
coarse keys and fine keys adding up to the targets, the `(C, D)` pair is returned.
-/
import Mathlib
import CensusCore.Search

namespace GoethalsSeidel.Census

open CensusCore Std

theorem mem_groupBy_aux {α κ : Type} [BEq κ] [Hashable κ] [LawfulBEq κ] (key : α → κ)
    (l : List α) (m : HashMap κ (List α)) (a : α)
    (h : a ∈ l ∨ a ∈ m.getD (key a) []) :
    a ∈ (l.foldl (fun m a => m.insert (key a) (a :: m.getD (key a) [])) m).getD (key a) [] := by
  induction l generalizing m with
  | nil => simpa using h
  | cons x xs ih =>
    simp only [List.foldl_cons]
    apply ih
    rcases h with h | h
    · rcases List.mem_cons.mp h with rfl | h
      · right; rw [HashMap.getD_insert]; simp
      · left; exact h
    · right
      rw [HashMap.getD_insert]
      split_ifs with hk
      · have : key x = key a := by simpa using hk
        rw [this]; exact List.mem_cons_of_mem _ h
      · exact h

theorem mem_groupBy {α κ : Type} [BEq κ] [Hashable κ] [LawfulBEq κ] (key : α → κ)
    {l : List α} {a : α} (h : a ∈ l) : a ∈ (groupBy key l).getD (key a) [] :=
  mem_groupBy_aux key l _ a (Or.inl h)

theorem exists_mem_toList {κ β : Type} [BEq κ] [Hashable κ] [LawfulBEq κ]
    {m : HashMap κ (List β)} {k : κ} {b : β} (h : b ∈ m.getD k []) :
    ∃ l, (k, l) ∈ m.toList ∧ b ∈ l := by
  rw [HashMap.getD_eq_getD_getElem?] at h
  cases hk : m[k]? with
  | none => rw [hk] at h; simp at h
  | some l =>
    rw [hk] at h
    exact ⟨l, HashMap.mem_toList_iff_getElem?_eq_some.mpr hk, h⟩

theorem join_complete {T : K2} {T' : K3} {LA LB LC LD : List RowData} {a b c d : RowData}
    (ha : a ∈ LA) (hb : b ∈ LB) (hc : c ∈ LC) (hd : d ∈ LD)
    (hck : add2 (add2 a.ck b.ck) (add2 c.ck d.ck) = T)
    (hfk : add3 (add3 a.fk b.fk) (add3 c.fk d.fk) = T') :
    (c.v, d.v) ∈ join T T' LA LB LC LD := by
  have hK : sub2 T (add2 c.ck d.ck) = add2 a.ck b.ck := by
    subst hck; simp only [add2, sub2, Prod.mk.injEq]; constructor <;> ring
  have hF : sub3 T' (add3 c.fk d.fk) = add3 a.fk b.fk := by
    subst hfk; simp only [add3, sub3, Prod.mk.injEq]; refine ⟨by ring, by ring, by ring⟩
  obtain ⟨la, hla, hal⟩ := exists_mem_toList (mem_groupBy (fun r : RowData => r.ck) ha)
  obtain ⟨lb, hlb, hbl⟩ := exists_mem_toList (mem_groupBy (fun r : RowData => r.ck) hb)
  obtain ⟨lc, hlc, hcl⟩ := exists_mem_toList (mem_groupBy (fun r : RowData => r.ck) hc)
  obtain ⟨ld, hld, hdl⟩ := exists_mem_toList (mem_groupBy (fun r : RowData => r.ck) hd)
  unfold join
  simp only
  set gA := (groupBy (fun r : RowData => r.ck) LA).toList
  set gB := (groupBy (fun r : RowData => r.ck) LB).toList
  set gC := (groupBy (fun r : RowData => r.ck) LC).toList
  set gD := (groupBy (fun r : RowData => r.ck) LD).toList
  set abC := gA.flatMap fun ka => gB.map fun kb => (add2 ka.1 kb.1, (ka.2, kb.2)) with habC
  set cdC := gC.flatMap fun kc => gD.map fun kd => (sub2 T (add2 kc.1 kd.1), (kc.2, kd.2))
    with hcdC
  have heab : (add2 a.ck b.ck, (la, lb)) ∈ abC := by
    rw [habC, List.mem_flatMap]
    exact ⟨(a.ck, la), hla, List.mem_map.mpr ⟨(b.ck, lb), hlb, rfl⟩⟩
  have hecd : (add2 a.ck b.ck, (lc, ld)) ∈ cdC := by
    rw [hcdC, List.mem_flatMap, ← hK]
    exact ⟨(c.ck, lc), hlc, List.mem_map.mpr ⟨(d.ck, ld), hld, rfl⟩⟩
  obtain ⟨gl, hgl, hegl⟩ := exists_mem_toList
    (mem_groupBy (fun e : K2 × (List RowData × List RowData) => e.1) heab)
  have hcds := mem_groupBy (fun e : K2 × (List RowData × List RowData) => e.1) hecd
  simp only at hgl hegl hcds
  rw [List.mem_flatMap]
  refine ⟨(add2 a.ck b.ck, gl), hgl, ?_⟩
  simp only
  have hne : (HashMap.getD (groupBy (fun e : K2 × (List RowData × List RowData) => e.1) cdC)
      (add2 a.ck b.ck) []).isEmpty = false := by
    rcases hx : HashMap.getD (groupBy (fun e : K2 × (List RowData × List RowData) => e.1) cdC)
      (add2 a.ck b.ck) [] with _ | ⟨y, ys⟩
    · rw [hx] at hcds; simp at hcds
    · rfl
  rw [if_neg (by rw [hne]; simp)]
  rw [List.mem_flatMap]
  refine ⟨(add2 a.ck b.ck, (lc, ld)), hcds, ?_⟩
  rw [List.mem_flatMap]
  refine ⟨c, hcl, ?_⟩
  rw [List.mem_map]
  refine ⟨d, ?_, rfl⟩
  rw [List.mem_filter]
  refine ⟨hdl, ?_⟩
  rw [hF, HashSet.contains_ofList, List.contains_iff_mem, List.mem_flatMap]
  refine ⟨(add2 a.ck b.ck, (la, lb)), hegl, ?_⟩
  rw [List.mem_flatMap]
  exact ⟨a, hal, List.mem_map.mpr ⟨b, hbl, rfl⟩⟩

end GoethalsSeidel.Census
